#ifndef STC_EMBED_C_H
#define STC_EMBED_C_H

#include "common.h"
/* Inputs:
	cover - the binary cover vector
	coverlength - length of the cover vector
	message - the binary message to be hidden
	messagelength - length of the message
	profile - vector of finite, nonnegative IEEE-754 binary64 change costs
	stego - pointer to an array of length 'coverlength' to receive the stego message; this parameter can be NULL
	constr_height - the constraint height of the matrix; a higher value increases
	state, path, and time costs and may reduce finite-length coding loss

Return value:
	On success, the function returns the total distortion introduced by the embedding.
	On failure, the function throws stc_exception; it does not encode failure as
	a numeric distortion.
*/

double stc_embed(const u8 *cover, int coverlength, const u8 *message,
        int messagelength, const double *profile, u8 *stego,
        int constr_height = 10);

#endif
