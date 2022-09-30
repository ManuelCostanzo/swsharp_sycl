/*
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

Contact the author by mkorpar@gmail.com.
*/
/**
@file

@brief Provides object for storing named sequnces.
*/

#ifndef __SW_SHARP_CHAINH__
#define __SW_SHARP_CHAINH__

#ifdef __cplusplus 
extern "C" {
#endif

/*!
@brief Chain object used for storing named sequnces.

Chain object is created from a named sequences and is used for alignment. 
Chain characters are coded with scorerEncode(char) function. Chain object
provides view method for creating subchains and reverse subchains in constant 
time. On that behalf every chain object uses approximately 2 times more memory 
than the input sequence. 
*/
typedef struct Chain Chain;

/*!
@brief Chain object constructor.

Method constructs the chain object with a given name and character sequence. 
All given data is copied. Any non-alphabetical characters in the sequence are 
ignored. All characters in the sequence are changed to uppercase.

@param name chain name
@param nameLen chain name length
@param string chain characters
@param stringLen chain characters length

@return chain object 
*/
extern Chain* chainCreate(char* name, int nameLen, char* string, int stringLen);

/*!
@brief Chain destructor.

@param chain chain object 
*/
extern void chainDelete(Chain* chain);

/*!
@brief Chain char getter.

Method retrives the char from the index position. Index must be greater or equal 
to zero and less than chain length.

@param chain chain object 
@param index chain char index

@return chain char
*/
extern char chainGetChar(Chain* chain, int index);

/*!
@brief Chain code getter.

Method retrives the code from the index position. Index must be greater or equal 
to zero and less than chain length. Chain caracter is coded with the 
scorerEncode(char) function.

@param chain chain object 
@param index chain code index

@return chain code
*/
extern char chainGetCode(Chain* chain, int index);

/*!
@brief Chain length getter.

@param chain chain object 

@return length
*/
extern int chainGetLength(Chain* chain);

/*!
@brief Chain name getter.

@param chain chain object 

@return name
*/
extern const char* chainGetName(Chain* chain);

/*!
@brief Chain codes getter.

@param chain chain object 

@return chain codes
*/
extern const char* chainGetCodes(Chain* chain);

/*!
@brief Creates a view to the chain object.

Method creates a subchain from the input chain. It doesn't copy the data since
the original chain object is immutable. However the top parent chain object
(one created via the constructor), should not be deleted before the views are 
deleted. Both the start and end indexes shoulde be greater than 1 and less than 
the chain length.

@param chain chain object
@param start start index, inclusive
@param end end index, inclusive
@param reverse bool, 1 if the view should be reverse 0 otherwise

@return chain view 
*/
extern Chain* chainCreateView(Chain* chain, int start, int end, int reverse);

/*!
@brief Copies chain code to a buffer.

Destination buffer should be allocated and its length should be greater or equal
to chain length.

@param chain chain object
@param dest pointer to the destination buffer
*/
extern void chainCopyCodes(Chain* chain, char* dest);

/*!
@brief Chain deserialization method.

Method deserializes chain object from a byte buffer.

@param bytes byte buffer

@return chain object
*/
extern Chain* chainDeserialize(char* bytes);

/*!
@brief Chain serialization method.

Method serializes chain object to a byte buffer.

@param bytes output byte buffer
@param bytesLen output byte buffer length
@param chain chain object
*/
extern void chainSerialize(char** bytes, int* bytesLen, Chain* chain);

#ifdef __cplusplus 
}
#endif
#endif // __SW_SHARP_CHAINH__
