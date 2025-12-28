# Evil Hangman

Evil Hangman is a deceptive variation of the classic Hangman word game.  
Unlike the traditional version, the computer does not commit to a single secret word at the beginning. Instead, it dynamically adapts its word choice after each guess to avoid revealing letters for as long as possible.

This project is implemented in **C++ (C++23)** and built using **GNU Make**.

---

## Overview

In standard Hangman, the secret word is fixed from the start.  
In Evil Hangman:

- The game begins with a large set of possible words
- After each player guess, the set is partitioned based on how the guessed letter appears
- The largest partition that reveals the least information is selected
- The secret word is only finalized when no alternative remains

This strategy makes the game significantly more challenging.

## Dependencies

To build and run this project, the following tools are required:

### Required
- **GNU g++** (version supporting C++23)
- **GNU Make**
- **Standard C++ library**

### Optional
- **ccache** (used automatically if installed, speeds up recompilation)
- **valgrind** (for memory leak and error checking)

## How to Use

### Starting the Game

Run the program from the terminal:

```bash
./build/TEST