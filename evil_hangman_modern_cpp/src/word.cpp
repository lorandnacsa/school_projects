#include "word.h"

Word::Word(std::string t) : text(std::move(t)) {}

std::string Word::make_mask(char guess, const std::string &cur) const {
    std::string p = cur;
    for(size_t i = 0; i < text.size(); ++i)
        if(text[i] == guess)
            p[i] = guess;
    return p;
}

int Word::count_in_mask(const std::string &mask, char guess) const {
    int c = 0;
    for(char ch : mask)
        if(ch == guess) ++c;
    return c;
}

int Word::sum_pos(const std::string &mask, char guess) const {
    int s = 0;
    for(size_t i = 0; i < mask.size(); ++i)
        if(mask[i] == guess) s += (int)i;
    return s;
}
