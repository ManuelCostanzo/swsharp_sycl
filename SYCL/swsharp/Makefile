CC = g++ -march=native -O3

ifeq ($(BACK), icpx)
#SYCL_TARGETS=spir64,spir64_x86_64
	CP = icpx -fsycl -fsycl-unnamed-lambda -O3 -fsycl-targets=$(SYCL_TARGETS) -DSYCL
endif

ifeq ($(BACK), intel)
#SYCL_TARGETS=nvptx64-nvidia-cuda,spir64
	CP = clang++ -fsycl -fsycl-targets=$(SYCL_TARGETS) -fsycl-unnamed-lambda -O3 -DSYCL
endif

ifeq ($(BACK), intel_amd)
#SYCL_TARGETS=spir64_x86_64,amdgcn-amd-amdhsa 
#SYCL_TARGET_BACKEND=backend=amdgcn-amd-amdhsa
#SYCL_TARGET_OFFLOAD=gfx1031
	CP = clang++ -fsycl -fsycl-targets=$(SYCL_TARGETS) -Xsycl-target-backend=$(SYCL_TARGET_BACKEND) --offload-arch=$(SYCL_TARGET_OFFLOAD) -O3  -DSYCL
endif

ifeq ($(BACK), acpp)
#SYCL_TARGETS=generic
#SYCL_TARGETS=cuda:sm_80
	CP = acpp --acpp-targets="$(SYCL_TARGETS)" -O3 -DHIP -DSYCL
endif

ifeq ($(BACK), fpga)
	CP = icpx -fsycl -fintelfpga -Xshardware -Xstarget=/opt/intel/oneapi/intel_a10gx_pac:pac_a10 -v  -DSYCL
endif

CU = ${CP} 
LD = ${CP} 


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
OBJ = $(subst $(SRC_DIR), $(OBJ_DIR), $(addsuffix _$(BIN_OUTPUT).o, $(basename $(SRC))))
DEP = $(OBJ:.o=.d)
INC = $(subst $(SRC_DIR), $(INC_DIR), $(API))
LIB = $(LIB_DIR)/lib$(NAME)_$(BIN_OUTPUT).a
EXC = $(NAME)
BIN = $(EXC_DIR)/$(EXC)_$(BIN_OUTPUT)
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

$(OBJ_DIR)/%_$(BIN_OUTPUT).o: $(SRC_DIR)/%.c
	@echo [CC] $<
	@mkdir -p $(dir $@)
	@$(CC) $< -c -o $@ -MMD $(CC_FLAGS)

$(OBJ_DIR)/%_$(BIN_OUTPUT).o: $(SRC_DIR)/%.cpp
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
	@if [ -d "$(OBJ_DIR)" ]; then find $(OBJ_DIR) -name '*_$(BIN_OUTPUT).*' -exec rm -rf {} +; fi
	@rm -rf $(EXC)

remove:
	@echo [RM] removing
	@if [ -f "$(LIB)" ]; then rm -rf $(LIB); fi
	@if [ -f "$(BIN)" ]; then rm -rf $(BIN); fi
	@if [ -f "$(EXC)" ]; then rm -rf $(EXC); fi


-include $(DEP)
