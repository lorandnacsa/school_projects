#ifndef TUI_H
#define TUI_H

#include <iostream>
#include <string>
#include <vector>

class Game;

class Tui {
private:
    Game& game;

public:
    Tui(Game& g);

    void clear_screen();
    void draw_hangman();
    void print_guesses();
    void run_terminal_ui();
};

#endif
