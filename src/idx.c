#include "idx.h"
#include <stdio.h>
#include <stdlib.h>

static unsigned int big_to_little_int(unsigned int num)
{

	return ((num>>24)&0xff) | // move byte 3 to byte 0
                    ((num<<8)&0xff0000) | // move byte 1 to byte 2
                    ((num>>8)&0xff00) | // move byte 2 to byte 1
                    ((num<<24)&0xff000000); // byte 0 to byte 3

}

idx_buffer *idx_read(char *path)
{

	FILE *fptr = fopen(path, "r");

	if (!fptr)
	{
		fprintf(stderr,"couldn't open file %s",path);
		return 0;
	}

	unsigned int magic_header = 0;

	fread(&magic_header,sizeof(unsigned int),1,fptr);

	if (magic_header&0xFFFF)
	{

		fprintf(stderr,"file magic header incorrect");
		return 0;

	}

	if ((magic_header>>16)&0xFF != 8)
	{

		fprintf(stderr,"parser only supports unsigned byte data");
		return 0;

	}

	size_t n_dimensions = (magic_header>>24)&0xFF;

	if (n_dimensions > 3)
	{

		fprintf(stderr,"parser only supports parsing of up to 2 dimensional matrices");
		return 0;

	}

	unsigned int samples = 0;
	unsigned int rows = 1;
	unsigned int cols = 1;

	if (n_dimensions > 0)
	{
		fread(&samples,sizeof(unsigned int),1,fptr);
		samples = big_to_little_int(samples);
	}

	if (!samples)
	{

		fprintf(stderr,"no samples in file");
		return 0;

	}

	if (n_dimensions > 1)
	{

		fread(&rows,sizeof(unsigned int),1,fptr);
		rows = big_to_little_int(rows);
	}

	if (n_dimensions > 2)
	{
		fread(&cols,sizeof(unsigned int),1,fptr);
		cols = big_to_little_int(cols);
	}

	if (rows == 0 || cols == 0)
	{

		fprintf(stderr,"matrix cannot have 0 dimensions");
		return 0;

	}

	idx_buffer *idx_buf = (idx_buffer *)malloc(sizeof(idx_buf));

	idx_buf->n_samples = samples;

	idx_buf->samples = (matrix **)malloc(sizeof(matrix *) * samples);

	printf("reading idx file %s:\n",path);

	for (int i = 0; i < samples; i++)
	{

		idx_buf->samples[i] = mat_create(rows,cols);

		unsigned char buf[rows*cols];

		fread(&buf,sizeof(unsigned char),rows*cols,fptr);

		for(int j = 0; j < rows*cols; j++)
		{

			idx_buf->samples[i]->data[j] = buf[j];

		}

		if (i % (samples / 100) == 0)
		{

			printf("%d%%\n", i * 100 / samples);

		}

	}

	fclose(fptr);

	return idx_buf;

}

// frees the idx internal structure and the idx buffer itself
void idx_free(idx_buffer *idx_buf)
{

	for (int i = 0; i < idx_buf->n_samples; i++)
		mat_free(idx_buf->samples[i]);

	free(idx_buf->samples);

	free(idx_buf);

}