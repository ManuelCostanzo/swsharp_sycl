CC = g++ -O3 -march=native 

ifeq ($(BACK), icpx)
	CP = icpx -fsycl -fsycl-unnamed-lambda -O3 -fsycl-targets=spir64_x86_64 ../lib/libswsharp.a -march=native
	CU = icpx -fsycl -fsycl-unnamed-lambda -O3 -fsycl-targets=spir64_x86_64 ../lib/libswsharp.a -march=native
	LD = icpx -fsycl -fsycl-unnamed-lambda -O3 -fsycl-targets=spir64_x86_64 ../lib/libswsharp.a -march=native
endif


ifeq ($(BACK), intel)
	 CP = clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64 -fsycl-unnamed-lambda -O3
	 CU = clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64 -fsycl-unnamed-lambda -O3
	 LD = clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64 -fsycl-unnamed-lambda -O3
endif

ifeq ($(BACK), amd)
	CP = clang++ -fsycl -fsycl-targets=spir64_x86_64,amdgcn-amd-amdhsa  -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx1031 -O3 ../lib/libswsharp.a
	CU = clang++ -fsycl -fsycl-targets=spir64_x86_64,amdgcn-amd-amdhsa  -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx1031 -O3 ../lib/libswsharp.a
	LD = clang++ -fsycl -fsycl-targets=spir64_x86_64,amdgcn-amd-amdhsa  -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx1031 -O3 ../lib/libswsharp.a
endif

ifeq ($(BACK), acpp)
	CP = acpp --acpp-targets="generic" -DSYCL_LANGUAGE_VERSION=1 -O3 -DHIP
	CU = acpp --acpp-targets="generic" -DSYCL_LANGUAGE_VERSION=1 -O3 -DHIP
	LD = acpp --acpp-targets="generic" -DSYCL_LANGUAGE_VERSION=1 -O3 -DHIP
endif


DX = doxygen

NAME = swsharpn

OBJ_DIR = obj
SRC_DIR = src
DOC_DIR = doc
INC_DIR = ../include/$(NAME)
LIB_DIR = ../lib
EXC_DIR = ../bin
#WIN_DIR = ../swsharpwin/$(NAME)

I_CMD = $(addprefix -I, $(SRC_DIR) ../include )
L_CMD = $(addprefix -L, ../lib )

DEP_LIBS = ../lib/libswsharp.a

CC_FLAGS = $(I_CMD) -O3 -Wall -march=native
CP_FLAGS = $(CC_FLAGS)
CU_FLAGS = $(I_CMD) -O3
LD_FLAGS = $(I_CMD) $(L_CMD) -lswsharp -lpthread -lm -lstdc++

API = $(addprefix $(SRC_DIR)/, )

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
	@rm -rf $(OBJ_DIR) $(EXC)

remove:
	@echo [RM] removing
	@rm -rf $(INC_DIR) $(LIB) $(BIN) $(EXC)

-include $(DEP)
