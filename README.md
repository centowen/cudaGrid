# cudaGrid Experimental code for aperture synthesis grdding in cuda.

Currently quite minimal: Allows continuum imaging (but not cleaning) of a
single pointing contained in a casa ms-file. Mostly made as a benchmark
comparison to CASA. Currently significantly outperforms CASA: a factor of 10 on
a nvidia 480 vs. CASA on a intel E5-2620.

Depends on stacker (can be found at github) github.com/centowen/stacker.
