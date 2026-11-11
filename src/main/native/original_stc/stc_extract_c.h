#ifndef STC_EXTRACT_C_H
#define STC_EXTRACT_C_H

#include "common.h"

/* Inputs:
	stego - the binary stego vector
	stegolength - the length of the stego vector
	message - pointer to an array of length 'messagelength' to receive the extracted message
	messagelength - the length of the embedded message; it must be at least constr_height
	constr_height - the constraint height of the matrix used for embedding the message

Return values:
	0 on success, -1 on error
*/

int stc_extract(const u8 *stego, int stegolength, u8 *message, int messagelength, int constr_height = 10);

#endif
