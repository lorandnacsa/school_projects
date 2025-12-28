# ==============================================
# Project configuration
# ==============================================
PROG = TEST

# directory structure
SRC_DIR = src
BUILD_DIR = build
RELEASE_DIR = release
DOCS_DIR = docs

# ==============================================
# Compiler configuration
# ==============================================
CXX = ccache g++
CXXFLAGS = -Wall -Werror -Wpedantic -g -std=c++23 
LDFLAGS =

# Debug configuration
DEBUG = 1
ifeq ($(DEBUG), 1)
CXXFLAGS += -g -O0
LDFLAGS += -g
else
CXXFLAGS += -O2
endif

# ==============================================
# File lists
# ==============================================
HEADERS = $(wildcard $(SRC_DIR)/*.hpp $(SRC_DIR)/*.h)
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(addprefix $(BUILD_DIR)/,$(notdir $(SOURCES:.cpp=.o)))
vpath %.cpp $(SRC_DIR)

# ==============================================
# Targets
# ==============================================
.PHONY: all clean release run

all: $(BUILD_DIR)/$(PROG)

# Link the final executable
$(BUILD_DIR)/$(PROG): $(OBJECTS)
	@echo "Linking -> $@"
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

# Compile source files into object files
$(BUILD_DIR)/%.o: %.cpp $(HEADERS) | $(BUILD_DIR)
	@echo "Compiling -> $<"
	$(CXX) -c $(CXXFLAGS) $< -o $@

# Create build directory if missing
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Create release build
release: DEBUG = 0
release: clean all
	@mkdir -p $(RELEASE_DIR)
	@cp $(BUILD_DIR)/$(PROG) $(RELEASE_DIR)/
	@echo "Release build created at $(RELEASE_DIR)/$(PROG)"

# Clean up
clean:
	rm -rf $(BUILD_DIR) $(RELEASE_DIR)/$(PROG)

# Run the program (builds if not built yet)
run: all
	./$(BUILD_DIR)/$(PROG)

VALGRIND = valgrind
VALGRIND_FLAGS = --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose

.PHONY: valgrind memcheck

# Run the program with Valgrind
valgrind: all
	$(VALGRIND) $(VALGRIND_FLAGS) ./$(BUILD_DIR)/$(PROG)

# Shortcut alias
memcheck: valgrind


#.PHONY: release
#release:
#	rm -rf $(RELEASE_DIR)
#	mkdir $(RELEASE_DIR)
#	cp -rf $(HEADERS) $(SOURCES) $(RELEASE_DIR)/
#	cp -rf $(DOCS_DIR)/*.pdf $(RELEASE_DIR)/
#	7z a $(RELEASE_DIR)/NEPTUN.zip $(RELEASE_DIR)/*
#	7z l $(RELEASE_DIR)/NEPTUN.zip

#.PHONY: clean
#clean:
#	rm -rf $(BUILD_DIR) $(RELEASE_DIR)
