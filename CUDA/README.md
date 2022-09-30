# SW# 

SW# (swsharp) is a library for sequence alignment based on CUDA enabled GPUs. It utilizes Hirschbergs and Ukkonens algorithm for memory efficiency and additional speed up. The library is scalable for use with multiple GPUs. Some parts of the library utilize MPI for CUDA enabled clusters. In database searches library utilizes swimd, a SSE4 sequence alignment library.

This work has been supported in part by Croatian Science Foundation under the project UIP-11-2013-7353.

## DEPENDENCIES

### WINDOWS

1. CUDA SDK 5.0
2. Visual Studio 2010 (optional)*

\*note: The Visual Studio project provided by this distribution is only tested with the listed software. There is no guarantee it could be run on other software setups.

### LINUX and MAC OS

Application uses following software:

1. gcc 4.*+
2. nvcc 2.*+
3. doxygen - for documentation generation (optional)
4. graphviz - for documentation generation (optional)
5. mpi - for swsharpdbmpi (optional)

## INSTALLATION

### LINUX and MAC OS
Makefile is provided in the project root folder. If mpi is available uncomment the swsharpdbmpi module on the top of the Makefile. After running make and all dependencies are satisfied, include, lib and bin folders will appear. All executables are located in the bin folder. Exposed swsharp core api is located in the include folder, and swsharp core static library is found in the lib folder. An example of using the library can be seen in swsharpn module.

### WINDOWS
Download the Visual Studio project from https://sourceforge.net/projects/swsharp/files/. Swsharp project is set up as a static library and can be used by additional modules by linking it.

### MODULES

Currently supported modules are:

1. swsharp - Module is the main static library used by other modules and does not provide any executable.
2. swsharpn - Module is used for aligning nucleotide sequnces.
3. swsharpp - Module is used for aligning protein sequnces.
4. swsharpnc - Module is used for aligning which searches the best scores on both strands of a nucleotide sequnces.
5. swsharpdb - Module is used for aligning two protein sequence databases.

## EXAMPLES

All examples persume the make command from the project root folder was executed.

### Executables

Simple align of pair of nucleotides in fasta format can be executed on linux platforms from the project root folder with the command:

    ./bin/swsharpn -i examples/NC_000898.fasta -j examples/NC_007605.fasta

Simple protein fasta database search can be executed on linux platforms from the project root folder with the command:

    ./bin/swsharpdb -i examples/P18080.fasta -j examples/uniprot_sprot_small.fasta

\*note: First swsharpdb run for every target database will cache the database for future usage and therefore will be slower (file with .swsharp extension will be created). Next runs with the same target database will use the cached database and will be faster.

### Library

Simple pairwise alignment library usage can be seen in the following simple.c file. This short program aligns two pair of nucleotides in fasta format. The nucleotides paths are read from the command line as the first two arguments. This examples is for the linux platform.

simple.c:
    
    #include "swsharp/swsharp.h"

    int main(int argc, char* argv[]) {
    
        Chain* query = NULL;
        Chain* target = NULL; 
        
        // read the query as the first command line argument
        readFastaChain(&query, argv[1]);
        
        // read the target as the first command line argument
        readFastaChain(&target, argv[2]);
        
        // use one CUDA card with index 0
        int cards[] = { 0 };
        int cardsLen = 1;
        
        // create a scorer object
        // match = 1
        // mismatch = -3
        // gap open = 5
        // gap extend = 2
        Scorer* scorer;
        scorerCreateScalar(&scorer, 1, -3, 5, 2);
    
        // do the pairwise alignment, use Smith-Waterman algorithm
        Alignment* alignment;
        alignPair(&alignment, SW_ALIGN, query, target, scorer, cards, cardsLen, NULL);
         
        // output the results in emboss stat-pair format
        outputAlignment(alignment, NULL, SW_OUT_STAT_PAIR);
        
        // clean the memory
        alignmentDelete(alignment);
    
        chainDelete(query);
        chainDelete(target);
        
        scorerDelete(scorer);
        
        return 0;
    }

This code can be compiled with:

    nvcc simple.c -I include/ -L lib/ -l swsharp -l pthread -o simple

And the executable can be run with:
    
    ./simple input1.fasta input2.fasta

To build the API documentation, enter the swsharp folder and run the command:

    make docs
    
To view the documentation open the file {project_root}/swsharp/doc/html/index.html in a web browser.

## NOTES

Individual README files for executables are available in folders of the same name as the executable. 
