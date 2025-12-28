#ifndef GAME_H
#define GAME_H

#include <vector>
#include <string>
#include <memory>
#include "word.h"

class Tui;

class Game {
    private:
        std::vector<std::unique_ptr<Word>> all;
        std::vector<Word*> candidates;
        std::string mask;
        std::vector<char> wrong_guesses;
        std::vector<char> all_guesses;
        int remaining;
        int max_wrong;
        std::unique_ptr<Tui> tui;

        void load_dictionary(const std::string &path);

    public:
        Game();
        Game(const std::string &dict_path, int word_len, int max_wrong_guesses = 8);
        void start();
        bool guess(char g);
        std::string get_display_word() const;
        std::vector<char> get_wrong_guesses() const;
        std::vector<char> get_all_guesses() const;
        bool is_over() const;
        bool is_won() const;
        int get_max_wrong() const;
        std::string reveal_solution() const;
};

#endif
