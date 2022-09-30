/*
This code represents a SYCL-compatible, DPC++-based version of SW#.
Copyright (C) 2022 Manuel Costanzo, contributor Enzo Rucci.

swsharp - CUDA parallelized Smith Waterman with applying Hirschberg's and
Ukkonen's algorithm and dynamic cell pruning.
Copyright (C) 2013 Matija Korpar, contributor Mile Šikić

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Contact SW# author by mkorpar@gmail.com.

Contact SW#-SYCL authors by mcostanzo@lidi.info.unlp.edu.ar, erucci@lidi.info.unlp.edu.ar
*/
/**
@file

@brief Provides object used for alignment scoring.
*/

#ifndef __SWSHARP_SCORERH__
#define __SWSHARP_SCORERH__

#ifdef __cplusplus
extern "C"
{
#endif

    /*!
    @brief Scorer object used for alignment scoring.

    Scorer is organized as a similarity table with additional gap penalties. Affine
    gap penalty model is used. Scorer codes are defined as characters. Scorer works
    only on input characters coded with the scorerEncode(char) method.

    */
    typedef struct Scorer Scorer;

    /*!
    @brief Scorer constructor.

    Input scores table must be an array which length is equal to maxCode * maxCode.
    Input scores table must be organized so that columns and rows correspond to the
    codes shown in scorerEncode(char).

    @param name scorer name, copy is made
    @param scores similarity table, copy is made
    @param maxCode maximum code that scorer should work with
    @param gapOpen gap open penalty given as a positive integer
    @param gapExtend gap extend penalty given as a positive integer

    @return scorer object
    */
    extern Scorer *scorerCreate(const char *name, int *scores, char maxCode,
                                int gapOpen, int gapExtend);

    /*!
    @brief Scorer destructor.

    @param scorer scorer object
    */
    extern void scorerDelete(Scorer *scorer);

    /*!
    @brief Gap extend penalty getter.

    Gap extend penalty is defined as a positive integer.

    @param scorer scorer object

    @return gap extend penalty
    */
    extern int scorerGetGapExtend(Scorer *scorer);

    /*!
    @brief Gap open penalty getter.

    Gap open penalty is defined as a positive integer.

    @param scorer scorer object

    @return gap open penalty
    */
    extern int scorerGetGapOpen(Scorer *scorer);

    /*!
    @brief Max code getter.

    Max code is defined as the highest code scorer can work with.

    @param scorer scorer object

    @return max code
    */
    extern char scorerGetMaxCode(Scorer *scorer);

    /*!
    @brief Max score getter.

    Max score is defined as the highest score two codes can be scored.

    @param scorer scorer object

    @return max score
    */
    extern int scorerGetMaxScore(Scorer *scorer);

    /*!
    @brief Name getter.

    Scorer name usually coresponds to similarity matrix names.

    @param scorer scorer object

    @return name
    */
    extern const char *scorerGetName(Scorer *scorer);

    /*!
    @brief Table getter.

    @param scorer scorer object

    @return table
    */
    extern const int *scorerGetTable(Scorer *scorer);

    /*!
    @brief Scalar getter.

    Getter for scalar property. Scorer is scalar if the similarity matrix can be
    reduced to match, mismatch scorer. In other words scorer is scalar if every two
    equal codes are scored equaly and every two unequal codes are scored equaly.

    @param scorer scorer object

    @return 1 if scorer if scalar 0 otherwise
    */
    extern int scorerIsScalar(Scorer *scorer);

    /*!
    @brief Scores two codes.

    Given scorer scores two given codes. Both codes should be greater or equal to 0
    and less than maxCode.

    @param scorer scorer object
    @param a first code
    @param b second code

    @return similarity score of a and b
    */
    extern int scorerScore(Scorer *scorer, char a, char b);

    /*!
    @brief Scorer deserialization method.

    Method deserializes scorer object from a byte buffer.

    @param bytes byte buffer

    @return scorer object
    */
    extern Scorer *scorerDeserialize(char *bytes);

    /*!
    @brief Scorer serialization method.

    Method serializes scorer object to a byte buffer.

    @param bytes output byte buffer
    @param bytesLen output byte buffer length
    @param scorer scorer object
    */
    extern void scorerSerialize(char **bytes, int *bytesLen, Scorer *scorer);

    /*!
    @brief Scorer static decoding method.

    Function is exact inverse of scorerEncode(char).

    @param c input character

    @return decoded character
    */
    extern char scorerDecode(char c);

    /*!
    @brief Scorer static encoding method.

    Encoding method is case insensitive. Function returns character code which is
    grater or equal to zero or if the codes are not available -1.

    Encoding is done in the following way:
        - characters 'A'-'Z' are encoded to 0-25
        - characters 'a'-'z' are encoded to 0-25
        - all other input characters cannot be encoded

    @param c input character

    @return encoded character or -1 if coding isn't available
    */
    extern char scorerEncode(char c);

#ifdef __cplusplus
}
#endif
#endif // __SWSHARP_SCORERH__
