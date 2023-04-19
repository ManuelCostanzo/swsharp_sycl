
INS_DIR = /usr

# uncomment swsharpdbmpi module if mpi is available 
CORE = swsharp
MODULES = swsharpn swsharpdb #swsharpdb swsharpdbmpi

INC_DIR = include/$(CORE)
LIB_DIR = lib
BIN_DIR = bin

INC_SRC = $(INC_DIR)
LIB_SRC = $(LIB_DIR)/lib$(CORE).a
BIN_SRC = $(addprefix $(BIN_DIR)/, $(MODULES))

INC_DST = $(addprefix $(INS_DIR)/, $(INC_SRC))
LIB_DST = $(addprefix $(INS_DIR)/, $(LIB_SRC))
BIN_DST = $(addprefix $(INS_DIR)/, $(BIN_SRC))

all: TARGETS=install
debug: TARGETS=debug install
cpu: TARGETS=cpu install
win: TARGETS=win
clean: TARGETS=remove clean
install: TARGETS=install

all: $(CORE) $(MODULES)

debug: $(CORE) $(MODULES)

cpu: $(CORE) $(MODULES)

win: $(CORE) $(MODULES)

clean: $(CORE) $(MODULES)
	@echo [RM] removing
	@rm -rf $(INC_DIR) $(LIB_DIR) $(EXC_DIR)

install: $(CORE) $(MODULES) $(INC_DST) $(LIB_DST) $(BIN_DST)

uninstall:
	@echo [RM] uninstalling
	@rm -rf $(INC_DST) $(LIB_DST) $(BIN_DST)

$(INS_DIR)/%: %
	@echo [CP] $@
	@mkdir -p $(dir $@)
	@cp -r $< $@

$(CORE): 
	@echo [CORE] $@
	@$(MAKE) -s -C $@ $(TARGETS)

$(MODULES): $(CORE)
	@echo [MOD] $@
	@$(MAKE) -s -C $@ $(TARGETS)

.PHONY: $(CORE) $(MODULES)
