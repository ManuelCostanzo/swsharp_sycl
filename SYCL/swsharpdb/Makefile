CC = g++  -march=native -O3 

DEP_LIBS = ../lib/libswsharp_$(BIN_OUTPUT).a

ifeq ($(BACK), icpx)
	CP = icpx -fsycl -fsycl-unnamed-lambda -O3 -fsycl-targets=$(SYCL_TARGETS) $(DEP_LIBS) -DSYCL
endif

ifeq ($(BACK), intel)
	CP = clang++ -fsycl -fsycl-targets=$(SYCL_TARGETS) -fsycl-unnamed-lambda -O3 -DSYCL
endif

ifeq ($(BACK), intel_amd)
	CP = clang++ -fsycl -fsycl-targets=$(SYCL_TARGETS) -Xsycl-target-backend=$(SYCL_TARGET_BACKEND) --offload-arch=$(SYCL_TARGET_OFFLOAD) -O3 $(DEP_LIBS) -DSYCL
endif

ifeq ($(BACK), acpp)
	CP = acpp --acpp-targets="$(SYCL_TARGETS)" -O3 -DHIP -DSYCL
endif

ifeq ($(BACK), fpga)
	CP = icpx -fsycl -fintelfpga -v $(DEP_LIBS) -DSYCL
endif

CU = ${CP} 
LD = ${CP}

DX = doxygen

NAME = swsharpdb

OBJ_DIR = obj
SRC_DIR = src
DOC_DIR = doc
INC_DIR = ../include/$(NAME)
LIB_DIR = ../lib
EXC_DIR = ../bin
#WIN_DIR = ../swsharpwin/$(NAME)

I_CMD = $(addprefix -I, $(SRC_DIR) ../include )
L_CMD = $(addprefix -L, ../lib )


CC_FLAGS = $(I_CMD) -O3 -Wall -march=native
CP_FLAGS = $(CC_FLAGS)
CU_FLAGS = $(I_CMD) -O3
LD_FLAGS = $(I_CMD) $(L_CMD) -lswsharp_$(BIN_OUTPUT) -lpthread -lm -lstdc++

API = $(addprefix $(SRC_DIR)/, )

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

all: $(EXC)
debug: all
cpu: all

install: bin #win

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
	@ar rcs $(LIB) $(OBJ)

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
