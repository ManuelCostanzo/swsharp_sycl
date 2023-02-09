CC = g++ -march=native -O3

ifeq ($(BACK), icpx)
	CP = icpx -fsycl -fsycl-unnamed-lambda -O3
	CU = icpx -fsycl -fsycl-unnamed-lambda -O3
	LD = icpx -fsycl -fsycl-unnamed-lambda -O3
endif


ifeq ($(BACK), intel)
	 CP = clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64 -fsycl-unnamed-lambda -O3
	 CU = clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64 -fsycl-unnamed-lambda -O3
	 LD = clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64 -fsycl-unnamed-lambda -O3
endif

ifeq ($(BACK), fpga)
	CP = icpx -fsycl -fintelfpga -Xshardware -Xstarget=/opt/intel/oneapi/intel_a10gx_pac:pac_a10 -v 
	CU = icpx -fsycl -fintelfpga -Xshardware -Xstarget=/opt/intel/oneapi/intel_a10gx_pac:pac_a10 -v 
	LD = icpx -fsycl -fintelfpga -Xshardware -Xstarget=/opt/intel/oneapi/intel_a10gx_pac:pac_a10 -v
endif


ifeq ($(BACK), hip)
	CP = syclcc --hipsycl-targets="omp;cuda:sm_75" -DSYCL_LANGUAGE_VERSION=1 -O3 -DHIP
	CU = syclcc --hipsycl-targets="omp;cuda:sm_75" -DSYCL_LANGUAGE_VERSION=1 -O3 -DHIP
	LD = syclcc --hipsycl-targets="omp;cuda:sm_75" -DSYCL_LANGUAGE_VERSION=1 -O3 -DHIP
endif


NAME = swsharp

OBJ_DIR = obj
SRC_DIR = src
DOC_DIR = doc
INC_DIR = ../include/$(NAME)
LIB_DIR = ../lib
EXC_DIR = ../bin
#WIN_DIR = ../swsharpwin/$(NAME)

I_CMD = $(addprefix -I, $(SRC_DIR) )
L_CMD = $(addprefix -L, )

DEP_LIBS =

CC_FLAGS = $(I_CMD) -O3 -Wall -march=native
CP_FLAGS = $(CC_FLAGS)
LD_FLAGS = $(I_CMD) $(L_CMD) -lpthread -lstdc++

CU_FLAGS = $(I_CMD) -O3

CU_FLAGS += #-gencode arch=compute_75,code=\"sm_75,compute_75\"

API = $(addprefix $(SRC_DIR)/, align.h alignment.h chain.h constants.h \
	cpu_module.h cuda_utils.h database.h db_alignment.h evalue.h gpu_module.h \
	post_proc.h pre_proc.h reconstruct.h scorer.h swsharp.h thread.h threadpool.h)

SRC = $(shell find $(SRC_DIR) -type f \( -iname \*.cpp -o -iname \*.c \))
HDR = $(shell find $(SRC_DIR) -type f \( -iname \*.h \))
OBJ = $(subst $(SRC_DIR), $(OBJ_DIR), $(addsuffix .o, $(basename $(SRC))))
DEP = $(OBJ:.o=.d)
INC = $(subst $(SRC_DIR), $(INC_DIR), $(API))
LIB = $(LIB_DIR)/lib$(NAME).a
EXC = $(NAME)
BIN = $(EXC_DIR)/$(EXC)
DOC = $(DOC_DIR)/Doxyfile
#WIN = $(subst $(SRC_DIR), $(WIN_DIR), $(HDR) $(SRC))

debug: CC_FLAGS := $(CC_FLAGS) -DDEBUG -DTIMERS
debug: CP_FLAGS := $(CP_FLAGS) -DDEBUG -DTIMERS
debug: CU_FLAGS := $(CU_FLAGS) -DDEBUG -DTIMERS --ptxas-options=-v

cpu: LD = $(CC)

all: $(OBJ) $(DEP_LIBS)
debug: all
cpu: all

install: lib include #win

bin: $(BIN)

include: $(INC)

lib: $(LIB)

#win: $(WIN)

$(EXC): $(OBJ) $(DEP_LIBS)
	@echo [LD] $@
	@mkdir -p $(dir $@)
	@$(LD) $(OBJ) -o $@ $(LD_FLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@echo [CC] $<
	@mkdir -p $(dir $@)
	@$(CC) $< -c -o $@ -MMD $(CC_FLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo [CP] $<
	@mkdir -p $(dir $@)
	@$(CP) $< -c -o $@ -MMD $(CP_FLAGS)

$(INC_DIR)/%.h: $(SRC_DIR)/%.h
	@echo [CP] $@
	@mkdir -p $(dir $@)
	@cp $< $@
	
$(LIB): $(OBJ)
	@echo [AR] $@
	@mkdir -p $(dir $@)
	@ar rcs $(LIB) $(OBJ) 2> /dev/null

$(BIN): $(EXC)
	@echo [CP] $@
	@mkdir -p $(dir $@)
	@cp $< $@

#$(WIN_DIR)/%: $(SRC_DIR)/%
#	@echo [CP] $@
#	@mkdir -p $(dir $@)
#	@cp $< $@

docs:
	@echo [DX] generating documentation
	@$(DX) $(DOC)
	
clean:
	@echo [RM] cleaning
	@rm -rf $(OBJ_DIR) $(EXC)

remove:
	@echo [RM] removing
	@rm -rf $(INC_DIR) $(LIB) $(BIN) $(EXC)

-include $(DEP)
