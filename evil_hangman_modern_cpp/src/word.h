#ifndef WORD_H
#define WORD_H

#include <string>

struct Word {
    std::string text;

    Word(std::string t);

    std::string make_mask(char guess, const std::string &cur) const;
    int count_in_mask(const std::string &mask, char guess) const;
    int sum_pos(const std::string &mask, char guess) const;
};

#endif
