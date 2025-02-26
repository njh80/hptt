#include <stdlib.h>

template<typename floatType>
void transpose_ref( uint32_t *size, uint32_t *perm, int dim, 
      const floatType* __restrict__ A, floatType alpha, int *outerSizeA, int *offsetA, 
      floatType* __restrict__ B, floatType beta, int *outerSizeB, int *offsetB, const bool conjA);
