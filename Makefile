# GCC compiler from MSYS2 UCRT64
# skriv mingw32-make i terminalen for at kompilere


CC = gcc

# Output file
TARGET = feature2_main.exe

# Source files
SRC = src/main.cpp src/feature_extractor.c

# Include folders
INCLUDES = -Isrc -I.

# Compiler flags
CFLAGS = -std=c11 -Wall -Wextra -O2

# Linker flags
LDFLAGS = -lm

# Build target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LDFLAGS)

# Run program
run: $(TARGET)
	.\$(TARGET)

# Clean generated files
clean:
	cmd /C del /Q $(TARGET) *.o *.obj 2>NUL