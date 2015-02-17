#include <cuda.h>
#include <iostream>
#include <fstream>
#include "ndarray.h"

#include "DataIO.h"
#include "msio.h"
#include "Chunk.h"
#include "Coords.h"
#include <definitions.h>
#include "error.h"
using std::endl;
using std::cout;
using std::ios;
using std::fstream;


// Definitions of constants./*{{{*/
const int N_STOKES = 2;
const int THREADS = 128;
const int BLOCKS = 128;
const int chunk_size = 100000;
const int MAX_PHASE_CENTRES = 100;

// cuda constants
__constant__ float field_omega_u[MAX_PHASE_CENTRES];
__constant__ float field_omega_v[MAX_PHASE_CENTRES];
__constant__ float field_omega_w[MAX_PHASE_CENTRES];
/*}}}*/

typedef struct _DataContainer/*{{{*/
{
	float* u;
	float* v;
	float* w;
	float* freq;
	float* data_real;
	float* data_imag;
	float* data_weight;
	int* spw;
	int* field;
} DataContainer;/*}}}*/

typedef struct _DataGrid/*{{{*/
{
	float* vis_real;
	float* vis_imag;
	float* weight;
	float cell;
	int nx, ny;
} DataGrid;/*}}}*/

// Function declarations./*{{{*/
__global__ void cudaGrid(DataContainer data, int chunk_size,
                         int nchan, DataGrid data_grid);
void grid(const string& vis, DataGrid& data_grid, int mode, float x0, float y0);
void grid_to_numpy_containers(const char* vis,
                              Ndarray<double, 2> vis_real,
                              Ndarray<double, 2> vis_imag,
                              Ndarray<double, 2> weight,
                              Ndarray<double, 3> pb,
                              double cell, float x0, float y0,
							  int mode);

extern "C"{
const int grid_mode_uniform = 0;
const int grid_mode_natural = 1;
const int grid_mode_uniform_unweighted = 2;
void c_grid(const char* vis,
            numpyArray<double> c_vis_real, numpyArray<double> c_vis_imag,
            numpyArray<double> c_weight,
            numpyArray<double> c_pb,
            const double cell, const float x0, const float y0, 
			int mode = grid_mode_natural)
{
    Ndarray<double, 2> vis_real(c_vis_real);
    Ndarray<double, 2> vis_imag(c_vis_imag);
    Ndarray<double, 2> weight(c_weight);
    Ndarray<double, 3> pb(c_pb);

	grid_to_numpy_containers(vis, vis_real, vis_imag, weight, pb, cell, x0, y0, mode);
};
}

void allocate_cuda_data(DataContainer& data, DataGrid data_grid,
                        int nx, int ny, float cell, int nchan,
                        int nstokes, int chunk_size);

void setup(DataIO*& dataio, string filein,
           DataContainer& dev_data,
		   DataGrid& dev_data_grid, DataGrid& data_grid,
		   float x0, float y0);
void setup_freq(DataContainer& data, DataIO* dataio);
void setup_grid(DataGrid& data_grid, int nx, int ny, float cell);
void setup_dev(DataGrid& data_grid, DataGrid& dev_data_grid,
		            DataIO* dataio, float x0, float y0);

int read_data_from_disk(DataIO* data, Chunk& chunk);
void copy_data_to_cuda(DataContainer& data, Chunk& chunk);
void copy_grid_from_cuda(DataGrid& data_grid, DataGrid& dev_data_grid);
void normalize_grid(DataGrid& data_grid, const int mode);
void write_grid_to_disk(DataGrid& data_grid, const string& gridded_data_file,
                        const string& uvcoverage_file);

void reset_data_grid(DataGrid& data_grid);
void delete_grid(DataGrid& data_grid);
void cleanup(DataIO*& dataio, DataContainer& dev_data,
             DataGrid& dev_data_grid);
void cleanup_freq(DataContainer& dev_data);
void cleanup_grid(DataGrid& dev_data_grid);
void free_cuda_data(DataContainer& data);/*}}}*/

int main(int argc, char* argv[])/*{{{*/
{
	string vis;
	string gridded_data_file;
	string uvcoverage_file;
	string mode_string;
	DataGrid data_grid;
	int mode;

	if(argc >= 4)
	{
		vis = argv[1];
		gridded_data_file = argv[2];
		uvcoverage_file = argv[3];
	}
	else
	{
		cerr << "grid vis outdatafile outuvcov [gridmode]" << endl;
	}

	if( argc >= 5)
	{
		if(string(argv[4]) == "natural")
		{
			mode = grid_mode_natural;
			cout << "Using natural weighting." << endl;
		}
		else if(string(argv[4]) == "uniform")
		{
			mode = grid_mode_uniform;
			cout << "Using uniform weighting." << endl;
		}
		else if(string(argv[4]) == "uniform_unweighted")
			mode = grid_mode_uniform_unweighted;
		else
			mode = grid_mode_natural;
	}
	else
	{
		mode = grid_mode_natural;
	}

	setup_grid(data_grid, 64, 64, 4.84813681109536e-06*0.2);
	grid(vis, data_grid, mode, 0., 0.);
	delete_grid(data_grid);
}/*}}}*/

__global__ void cudaGrid(DataContainer data, int chunk_size, int nchan,/*{{{*/
                         DataGrid data_grid, int nfields)
{
	int uvrow = threadIdx.x+blockIdx.x*blockDim.x;
	int grid_index, grid_index_inv;
	int u_index;
	int v_index;
	float weight_sum = 0.;
	float vis_real_sum = 0.;
	float vis_imag_sum = 0.;
	float phase_rot_phi;
	float phase_rot_real;
	float phase_rot_imag;

	while(uvrow < chunk_size && data.field[uvrow] == 0) // FIXME: Should check if field is being imaged.
	{
		float* freq = &data.freq[data.spw[uvrow]*nchan];
		for(int chanID = 0; chanID < nchan; chanID++)
		{
// 			u_index = int(data.u[uvrow]*freq[chanID]/c*data_grid.cell*data_grid.nx+data_grid.nx/2.);
// 			v_index = int(data.v[uvrow]*freq[chanID]/c*data_grid.cell*data_grid.ny+data_grid.ny/2.);
			u_index = int(-data.u[uvrow]*freq[chanID]/c*data_grid.cell*data_grid.nx+data_grid.nx/2.+0.5);
			v_index = int(data.v[uvrow]*freq[chanID]/c*data_grid.cell*data_grid.ny+data_grid.ny/2.+0.5);

			grid_index = u_index*data_grid.ny + v_index;
			grid_index_inv = (data_grid.nx-u_index)*data_grid.ny + (data_grid.ny-v_index);

			phase_rot_phi = -freq[chanID]*(data.u[uvrow]*(field_omega_u[data.field[uvrow]])+
			                               data.v[uvrow]*(field_omega_v[data.field[uvrow]])+
			                               data.w[uvrow]*(field_omega_w[data.field[uvrow]]));
// 			phase_rot_phi = 0.;
			sincos(phase_rot_phi, &phase_rot_imag, &phase_rot_real);
// 			phase_rot_real = cosf(phase_rot_phi);
// 			phase_rot_imag = sinf(phase_rot_phi);

// 			if(chanID == 0 and uvrow == 0 and data.spw[uvrow] == 0)
// 			{
// 				atomicAdd(&data_grid.vis_real[29+30*data_grid.nx],
// 				          phase_rot_phi);
// 				atomicAdd(&data_grid.vis_real[30+30*data_grid.nx],
// 				          phase_rot_real);
// 				atomicAdd(&data_grid.vis_real[31+30*data_grid.nx],
// 				          phase_rot_imag);
// 				atomicAdd(&data_grid.vis_real[32+30*data_grid.nx],
// 				          1.);
// 				atomicAdd(&data_grid.vis_real[33+30*data_grid.nx],
// 				          data.u[uvrow]);
// 				atomicAdd(&data_grid.vis_real[34+30*data_grid.nx],
// 				          field_omega_u[data.field[uvrow]]);
// 				atomicAdd(&data_grid.vis_real[35+30*data_grid.nx],
// 				          freq[chanID]);
// 				atomicAdd(&data_grid.vis_real[36+30*data_grid.nx],
// 				          field_omega_u[0]);
// 			}

			for(int stokesID=0; stokesID < N_STOKES; stokesID++)
			{
				int weightindex = uvrow*N_STOKES + stokesID;
				int dataindex = nchan*(weightindex) + chanID;

				atomicAdd(&data_grid.vis_real[grid_index],
				          data.data_weight[weightindex]*(data.data_real[dataindex]*phase_rot_real-
						                                 data.data_imag[dataindex]*phase_rot_imag));
				atomicAdd(&data_grid.vis_imag[grid_index],
				          data.data_weight[weightindex]*(data.data_real[dataindex]*phase_rot_imag+
						                                 data.data_imag[dataindex]*phase_rot_real));
				atomicAdd(&data_grid.weight[grid_index], data.data_weight[weightindex]);

				atomicAdd(&data_grid.vis_real[grid_index_inv],
				          data.data_weight[weightindex]*(data.data_real[dataindex]*phase_rot_real-
						                                 data.data_imag[dataindex]*phase_rot_imag));
				atomicAdd(&data_grid.vis_imag[grid_index_inv],
				          -1.*data.data_weight[weightindex]*(data.data_real[dataindex]*phase_rot_imag+
						                                     data.data_imag[dataindex]*phase_rot_real));
				atomicAdd(&data_grid.weight[grid_index_inv], data.data_weight[weightindex]);

				weight_sum += data.data_weight[weightindex];
				vis_real_sum += data.data_real[dataindex]*data.data_weight[weightindex];
				vis_imag_sum += data.data_imag[dataindex]*data.data_weight[weightindex];

// 				atomicAdd(&data_grid.vis_real[data_grid.nx*data_grid.ny], data.data_real[dataindex]*data.data_weight[weightindex]);
// 				atomicAdd(&data_grid.vis_imag[data_grid.nx*data_grid.ny], data.data_imag[dataindex]*data.data_weight[weightindex]);
// 				atomicAdd(&data_grid.weight[data_grid.nx*data_grid.ny], data.data_weight[weightindex]);
			}
		}
		uvrow+=blockDim.x*gridDim.x; // Update the index of each thread by the number of threads launched simultaneosly (threads per block * number of blocks).
	}
}/*}}}*/
void grid(const string& vis, DataGrid& data_grid,/*{{{*/
          const int mode, float x0, float y0)
{
	DataIO* dataio;
	DataContainer data;
	DataGrid dev_data_grid;
	Chunk chunk(chunk_size);

	setup(dataio, vis.c_str(), data, dev_data_grid, data_grid, x0, y0);

	while(read_data_from_disk( dataio,  chunk) > 0)
	{
		copy_data_to_cuda(data, chunk);
		cudaGrid<<<BLOCKS,THREADS>>>(data, chunk.size(), dataio->nChan(), dev_data_grid, dataio->nPointings());
		CudaCheckError();
		cout << "*" << std::flush;
	}
	cout << endl;

	copy_grid_from_cuda(data_grid, dev_data_grid);
	cout << "Done copying data back!" << endl;

	normalize_grid(data_grid, mode);
	cout << "Done normalizing data." << endl;


// 	cout.precision(10);
// 	cout.setf( std::ios::fixed, std:: ios::floatfield );

	cleanup(dataio, data, dev_data_grid);
}/*}}}*/

void allocate_cuda_data(DataContainer& data, const int nchan, const int nstokes, const int chunk_size)/*{{{*/
{
	CudaSafeCall(cudaMalloc( (void**)&data.u, sizeof(float)*chunk_size));
	CudaSafeCall(cudaMalloc( (void**)&data.v, sizeof(float)*chunk_size));
	CudaSafeCall(cudaMalloc( (void**)&data.w, sizeof(float)*chunk_size));
	CudaSafeCall(cudaMalloc( (void**)&data.data_real, sizeof(float)*chunk_size*nchan*nstokes));
	CudaSafeCall(cudaMalloc( (void**)&data.data_imag, sizeof(float)*chunk_size*nchan*nstokes));
	CudaSafeCall(cudaMalloc( (void**)&data.data_weight, sizeof(float)*chunk_size*nstokes));
	CudaSafeCall(cudaMalloc( (void**)&data.spw, sizeof(int)*chunk_size));
	CudaSafeCall(cudaMalloc( (void**)&data.field, sizeof(int)*chunk_size));
}/*}}}*/

void reset_data_grid(DataGrid& data_grid)/*{{{*/
{
	for(int i = 0; i < data_grid.nx*data_grid.ny+1; i++)
	{
		data_grid.vis_real[i] = 0.;
		data_grid.vis_imag[i] = 0.;
		data_grid.weight  [i] = 0.;
	}
}/*}}}*/

void setup_grid(DataGrid& data_grid, int nx, int ny, float cell)/*{{{*/
{
	data_grid.nx = nx;
	data_grid.ny = ny;
	data_grid.cell = cell;
	data_grid.vis_real = new float[nx*ny+1];
	data_grid.vis_imag = new float[nx*ny+1];
	data_grid.weight = new float[nx*ny+1];
	reset_data_grid(data_grid);
}/*}}}*/

void setup(DataIO*& dataio, string filein, /*{{{*/
           DataContainer& dev_data,
		   DataGrid& dev_data_grid, DataGrid& data_grid,
		   float x0, float y0)
{
	dataio = (DataIO*)new msio(filein.c_str(), "", true);

	allocate_cuda_data(dev_data, dataio->nChan(), N_STOKES, chunk_size);
	setup_freq(dev_data, dataio);
	setup_dev(data_grid, dev_data_grid, dataio, x0, y0);
// 	setup_grid(data_grid, dev_data_grid, 512, 512, 4.84813681109536e-06*0.2*0.5);
// 	setup_grid(data_grid, dev_data_grid, 4096, 4096, 4.84813681109536e-06*0.2*0.5);
}/*}}}*/
void setup_freq(DataContainer& data, DataIO* dataio)/*{{{*/
{
	float* freq = new float[dataio->nChan()*dataio->nSpw()];
	CudaSafeCall(cudaMalloc( (void**)&data.freq, sizeof(float)*dataio->nChan()*dataio->nSpw()));

	// Load frequencies into freq[].
	for(int chanID = 0; chanID < dataio->nChan(); chanID++)
	{
		for(int spwID = 0; spwID < dataio->nSpw(); spwID++)
		{
			freq[spwID*dataio->nChan()+chanID] = (float)dataio->getFreq(spwID)[chanID];
		}
	}
	CudaSafeCall(cudaMemcpy(data.freq, freq,
	           sizeof(float)*dataio->nChan()*dataio->nSpw(),
	           cudaMemcpyHostToDevice));
	delete[] freq;
}/*}}}*/
void setup_dev(DataGrid& data_grid, DataGrid& dev_data_grid, DataIO* dataio, float x0, float y0) /*{{{*/
{
	dev_data_grid.nx = data_grid.nx;
	dev_data_grid.ny = data_grid.ny;
	dev_data_grid.cell = data_grid.cell;
	CudaSafeCall(cudaMalloc( (void**)&dev_data_grid.vis_real,
	                         sizeof(float)*(dev_data_grid.nx*dev_data_grid.ny+1)));
	CudaSafeCall(cudaMalloc( (void**)&dev_data_grid.vis_imag,
	                         sizeof(float)*(dev_data_grid.nx*dev_data_grid.ny+1)));
	CudaSafeCall(cudaMalloc( (void**)&dev_data_grid.weight,
	                         sizeof(float)*(dev_data_grid.nx*dev_data_grid.ny+1)));
	CudaSafeCall(cudaMemcpy(dev_data_grid.vis_real, data_grid.vis_real,
	                        sizeof(float)*(data_grid.nx*data_grid.ny+1),
	                        cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_data_grid.vis_imag, data_grid.vis_imag,
	                        sizeof(float)*(data_grid.nx*data_grid.ny+1),
	                        cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_data_grid.weight, data_grid.weight,
	                        sizeof(float)*(data_grid.nx*data_grid.ny+1),
	                        cudaMemcpyHostToDevice));

	// Set up pointing information.
	int n_phase_centres = dataio->nPointings();
	if( n_phase_centres > MAX_PHASE_CENTRES)
	{
		std::cerr << "To many pointings for CUDA! Only taking " 
		          << n_phase_centres << " first." << endl;
		n_phase_centres = MAX_PHASE_CENTRES;
	}
	float *host_field_omega_u = new float[n_phase_centres];
	float *host_field_omega_v = new float[n_phase_centres];
	float *host_field_omega_w = new float[n_phase_centres];
	for(int i = 0; i < n_phase_centres; i++)
	{
		float dx = sin(x0 - dataio->xPhaseCentre(i)) * cos(y0);
		float dy = sin(y0)*cos(dataio->yPhaseCentre(i)) -
			       cos(y0)*sin(dataio->yPhaseCentre(i)) *
				   cos(x0-dataio->xPhaseCentre(i));
		dx = fmod(dx, 2*M_PI);
		host_field_omega_u[i] = 2*M_PI*dx/c;
		host_field_omega_v[i] = 2*M_PI*dy/c;
		host_field_omega_w[i] = 2*M_PI*(sqrt(1-dx*dx+dy*dy)-1)/c;
		if(i == 0)
		{
			cout << "(x0, y0) = " << x0 << ", " << y0 << endl;
			cout << "phase centre: " << dataio->xPhaseCentre(i) << ", " << dataio->yPhaseCentre(i) << endl;
			cout << "(dx, dy) = " << dx*180*3600/M_PI << ", " << dy*180*3600/M_PI << endl;
			cout << "(omega_u, omega_v) = " << host_field_omega_u[0] << ", " << host_field_omega_v[0] << endl;
		}
	}

	CudaSafeCall(cudaMemcpyToSymbol( field_omega_u, host_field_omega_u,
	                                 sizeof(float)*n_phase_centres, 0,
	                                 cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpyToSymbol( field_omega_v, host_field_omega_v,
	                                 sizeof(float)*n_phase_centres, 0,
	                                 cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpyToSymbol( field_omega_w, host_field_omega_w,
	                                 sizeof(float)*n_phase_centres, 0,
	                                 cudaMemcpyHostToDevice));

	delete[] host_field_omega_u;
	delete[] host_field_omega_v;
	delete[] host_field_omega_w;
}/*}}}*/

int read_data_from_disk(DataIO* data, Chunk& chunk)/*{{{*/
{
	int nrow = data->readChunk(chunk);
	return nrow;
}/*}}}*/
void copy_data_to_cuda(DataContainer& data, Chunk& chunk)/*{{{*/
{
	int chunk_size = chunk.size();
	float* u = new float[chunk_size];
	float* v = new float[chunk_size];
	float* w = new float[chunk_size];
	int *spw = new int[chunk_size];
	int *field = new int[chunk_size];
	for(int uvrow = 0; uvrow < chunk_size; uvrow++)
	{
		u[uvrow] = chunk.inVis[uvrow].u;
		v[uvrow] = chunk.inVis[uvrow].v;
		w[uvrow] = chunk.inVis[uvrow].w;
		spw[uvrow] = chunk.inVis[uvrow].spw;
		field[uvrow] = chunk.inVis[uvrow].fieldID;
	}

// 	cout << "u,v: " << u[0] << ", " << v[0] << endl;
// 	cout << "freq: " << chunk.inVis[0].freq[0] << endl;

	CudaSafeCall(cudaMemcpy(data.u, u, sizeof(float)*chunk.size(), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(data.v, v, sizeof(float)*chunk.size(), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(data.w, w, sizeof(float)*chunk.size(), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(data.spw, spw, sizeof(float)*chunk.size(), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(data.field, field, sizeof(float)*chunk.size(), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(data.data_real,   chunk.data_real_in,
				sizeof(float)*chunk.size()*chunk.nChan()*N_STOKES,
				cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(data.data_imag,   chunk.data_imag_in,
				sizeof(float)*chunk.size()*chunk.nChan()*N_STOKES,
				cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(data.data_weight, chunk.weight_in,
				sizeof(float)*chunk.size()*N_STOKES,
				cudaMemcpyHostToDevice));
	delete[] u;
	delete[] v;
	delete[] w;
	delete[] spw;
	delete[] field;
}/*}}}*/
void copy_grid_from_cuda(DataGrid& data_grid, DataGrid& dev_data_grid)/*{{{*/
{
	CudaSafeCall(cudaMemcpy(data_grid.vis_real, dev_data_grid.vis_real,
	                        sizeof(float)*(data_grid.nx*data_grid.ny+1),
	                        cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(data_grid.vis_imag, dev_data_grid.vis_imag,
	                        sizeof(float)*(data_grid.nx*data_grid.ny+1),
	                        cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(data_grid.weight, dev_data_grid.weight,
	                        sizeof(float)*(data_grid.nx*data_grid.ny+1),
	                        cudaMemcpyDeviceToHost));
}/*}}}*/
void normalize_grid(DataGrid& data_grid, const int mode)/*{{{*/
{
	if( mode == grid_mode_natural)
	{
		float sum_of_weights = 0.;
		for(int i = 0; i < data_grid.nx*data_grid.ny; i++)
				sum_of_weights += data_grid.weight[i];

		for(int i = 0; i < data_grid.nx*data_grid.ny; i++)
		{
			data_grid.vis_real[i] /= sum_of_weights;
			data_grid.vis_imag[i] /= sum_of_weights;
			data_grid.weight[i] /= sum_of_weights;
		}
	}
	else if( mode == grid_mode_uniform )
	{
		float sum_of_weights = 0.;
		for(int i = 0; i < data_grid.nx*data_grid.ny; i++)
		{
			if(data_grid.weight[i] > 0)
			{
				data_grid.vis_real[i] /= data_grid.weight[i];
				data_grid.vis_imag[i] /= data_grid.weight[i];
				data_grid.weight[i] = 1.;
				sum_of_weights += 1.;
			}
		}

		for(int i = 0; i < data_grid.nx*data_grid.ny; i++)
		{
			data_grid.vis_real[i] /= sum_of_weights;
			data_grid.vis_imag[i] /= sum_of_weights;
			data_grid.weight[i] /= sum_of_weights;
		}
	}
}/*}}}*/

void cleanup(DataIO*& dataio, DataContainer& dev_data,/*{{{*/
             DataGrid& dev_data_grid)
{
	cleanup_freq(dev_data);
	cleanup_grid(dev_data_grid);
	free_cuda_data(dev_data);
	delete dataio;
}/*}}}*/
void cleanup_freq(DataContainer& dev_data)/*{{{*/
{
	CudaSafeCall(cudaFree(dev_data.freq));
}/*}}}*/
void delete_grid(DataGrid& data_grid)/*{{{*/
{
	delete[] data_grid.vis_real;
	delete[] data_grid.vis_imag;
	delete[] data_grid.weight;
	data_grid.vis_real = NULL;
	data_grid.vis_imag = NULL;
	data_grid.weight = NULL;
}/*}}}*/
void cleanup_grid(DataGrid& dev_data_grid)/*{{{*/
{
	CudaSafeCall(cudaFree(dev_data_grid.vis_real));
	CudaSafeCall(cudaFree(dev_data_grid.vis_imag));
	CudaSafeCall(cudaFree(dev_data_grid.weight));
	dev_data_grid.vis_real = NULL;
	dev_data_grid.vis_imag = NULL;
	dev_data_grid.weight = NULL;
}/*}}}*/
void free_cuda_data(DataContainer& data)/*{{{*/
{
	CudaSafeCall(cudaFree( data.u));
	CudaSafeCall(cudaFree( data.v));
	CudaSafeCall(cudaFree( data.w));
	CudaSafeCall(cudaFree( data.data_real));
	CudaSafeCall(cudaFree( data.data_imag));
	CudaSafeCall(cudaFree( data.data_weight));
	CudaSafeCall(cudaFree( data.spw));
}/*}}}*/
void grid_to_numpy_containers(const char* vis, /*{{{*/
                              Ndarray<double, 2> vis_real,
                              Ndarray<double, 2> vis_imag,
                              Ndarray<double, 2> weight,
                              Ndarray<double, 3> pb,
                              double cell, float x0, float y0,
							  int mode)
{
	DataGrid data_grid;

	setup_grid(data_grid, vis_real.getShape(0), vis_real.getShape(1),
		       float(cell));

	string vis_tmp(vis);
// 	grid((string)vis, data_grid, mode);
	grid(vis_tmp, data_grid, mode, x0, y0);
// 	cout.precision(30);
// 	cout << "V(29,30): " << data_grid.vis_real[29+30*64] << endl;
// 	cout << "V(30,30): " << data_grid.vis_real[30+30*64] << endl;
// 	cout << "V(31,30): " << data_grid.vis_real[31+30*64] << endl;
// 	cout << "V(32,30): " << data_grid.vis_real[32+30*64] << endl;
// 	cout << "V(33,30): " << data_grid.vis_real[33+30*64] << endl;
// 	cout << "V(34,30): " << data_grid.vis_real[34+30*64] << endl;
// 	cout << "V(35,30): " << data_grid.vis_real[35+30*64] << endl;
// 	cout << "V(36,30): " << data_grid.vis_real[36+30*64] << endl;

	int len = vis_real.getShape(0)*vis_real.getShape(1);

    std::copy(data_grid.vis_real, &data_grid.vis_real[len], vis_real.begin());
    std::copy(data_grid.vis_imag, &data_grid.vis_imag[len], vis_imag.begin());
    std::copy(data_grid.weight, &data_grid.weight[len], weight.begin());
	delete_grid(data_grid);
}/*}}}*/
void write_grid_to_disk(DataGrid& data_grid, const string& gridded_data_file,/*{{{*/
		const string& uvcoverage_file)
{
	fstream datafile(gridded_data_file.c_str(), ios::out);
	fstream uvcovfile(uvcoverage_file.c_str(), ios::out);

	for(int uindex = 0; uindex < data_grid.nx; uindex++)
	{
		for(int vindex = 0; vindex < data_grid.ny; vindex++)
		{
			datafile << data_grid.vis_real[uindex+data_grid.nx*vindex];
			if(data_grid.vis_imag[uindex+data_grid.nx*vindex] >= 0.)
				datafile << "+";
			datafile << data_grid.vis_imag[uindex+data_grid.nx*vindex] << "j ";
		}
		datafile << "\n";
	}
	for(int uindex = 0; uindex < data_grid.nx; uindex++)
	{
		for(int vindex = 0; vindex < data_grid.ny; vindex++)
		{
			uvcovfile << data_grid.weight[uindex+data_grid.nx*vindex] << " ";
		}
		uvcovfile << "\n";
	}
	datafile.close();
	uvcovfile.close();
	cout << "Done writing data." << endl;
}/*}}}*/
