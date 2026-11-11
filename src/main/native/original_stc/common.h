#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <exception>
#include <string>
#include <utility>
#include <vector>

using u32 = std::uint32_t;
using u16 = std::uint16_t;
using u8 = std::uint8_t;

extern u32 mats[];

/* Simple class for throwing exceptions */
class stc_exception : public std::exception {
public:
    stc_exception(std::string message_value, u32 error_id_value)
            : error_id(error_id_value), message(std::move(message_value)) {}
    ~stc_exception() noexcept override = default;
    const char* what() const noexcept override { return message.c_str(); }
    u32 error_id;
private:
    std::string message;
};

/* 
   The following error_ids are in use:
   1 = Invalid STC argument or internal invariant.
   2 = Not enough memory.
   3 = The message cannot be longer than the cover object.
   4 = No solution exists.                                 - This happen when there are too many Inf values in cost vector and thus the solution does not exist due to sparse parity-check matrix.
   5 = Price vector limit exceeded.                        - There is a limit to cost elements when you use integer version of the algorithm. Try to use costs in double.
   6 = Maximum number of trials in layered construction exceeded.
 */

u32 *getMatrix(int width, int height);

/** Exact integer form of the public STC short/long submatrix schedule. */
struct stc_width_plan {
    int shorter;
    int longer;
    std::vector<int> matrix_choice;
    std::vector<int> widths;
};

int stc_shorter_width(int cover_length, int message_length);
int stc_longer_width(int cover_length, int message_length);
bool stc_should_use_longer(
        std::int64_t consumed,
        int message_index,
        int cover_length,
        int message_length);
stc_width_plan stc_make_width_plan(
        int cover_length, int message_length);

#endif
