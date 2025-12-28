#include "game.h"
#include "tui.h"
#include <fstream>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <unordered_set>

Game::Game() : remaining(0), max_wrong(8) {}

void Game::load_dictionary(const std::string &path) {
    std::ifstream ifs(path);
    if(!ifs) return;

    std::string w;
    while(ifs >> w) {
        std::string t;
        t.reserve(w.size());
        for(char c : w)
            if(std::isalpha((unsigned char)c))
                t.push_back((char)std::tolower((unsigned char)c));

        if(!t.empty())
            all.emplace_back(std::make_unique<Word>(std::move(t)));
    }
}

Game::Game(const std::string &dict_path, int word_len, int max_wrong_guesses)
    : remaining(0), max_wrong(max_wrong_guesses)
{
    if(!dict_path.empty()) load_dictionary(dict_path);

    if(all.empty()) {
        std::vector<std::string> sample = {
            "aroma","adoma","apuka","strand","start","zizeg","zokni","alma","banan",
            "korte","ablak","szek","lab","asztal","keret","program","akasztofavirag",
            "koporsoszeg","kotel","terminal","konyv"
        };
        for(auto &s : sample)
            all.emplace_back(std::make_unique<Word>(std::move(s)));
    }

    for(auto &u : all)
        if((int)u->text.size() == word_len)
            candidates.push_back(u.get());

    mask.assign(word_len, '.');
    remaining = max_wrong;
}

void Game::start() {
    tui = std::make_unique<Tui>(*this);
    tui->run_terminal_ui();
}

bool Game::guess(char g) {
    
    if(std::find(all_guesses.begin(), all_guesses.end(), g) != all_guesses.end())
        return false;

    all_guesses.push_back(g);

    std::unordered_map<std::string, std::vector<Word*>> families;

    for(Word* wp : candidates)
        families[wp->make_mask(g, mask)].push_back(wp);

    auto bestIt = families.begin();
    for(auto it = families.begin(); it != families.end(); ++it) {
        if(it == bestIt) continue;
        if(it->second.size() > bestIt->second.size()) { bestIt = it; continue; }
        if(it->second.size() < bestIt->second.size()) continue;

        int rIt = it->second.front()->count_in_mask(it->first, g);
        int rBest = bestIt->second.front()->count_in_mask(bestIt->first, g);
        if(rIt < rBest) { bestIt = it; continue; }
        if(rIt > rBest) continue;

        int sIt = it->second.front()->sum_pos(it->first, g);
        int sBest = bestIt->second.front()->sum_pos(bestIt->first, g);
        if(sIt > sBest) { bestIt = it; continue; }

        if(it->first < bestIt->first)
            bestIt = it;
    }

    std::vector<Word*> next = std::move(bestIt->second);
    std::string chosen_pattern = bestIt->first;

    std::unordered_set<Word*> keep(next.begin(), next.end());
    for(auto &up : all)
        if(up && !keep.count(up.get()))
            up.reset();

    all.erase(std::remove_if(all.begin(), all.end(),
            [](const std::unique_ptr<Word>&u){ return !u; }),
        all.end());
    all.shrink_to_fit();

    candidates = std::move(next);

    int newly = count_if(chosen_pattern.begin(), chosen_pattern.end(),
                         [&](char c){ return c == g; }) -
                count_if(mask.begin(), mask.end(),
                         [&](char c){ return c == g; });

    if(newly <= 0) {
        --remaining;
        wrong_guesses.push_back(g);
    }

    mask = std::move(chosen_pattern);
    return newly > 0;
}

std::string Game::get_display_word() const { return mask; }
std::vector<char> Game::get_wrong_guesses() const { return wrong_guesses; }
std::vector<char> Game::get_all_guesses() const { return all_guesses; }
bool Game::is_over() const { return remaining <= 0 || mask.find('.') == std::string::npos; }
bool Game::is_won() const { return mask.find('.') == std::string::npos; }
int Game::get_max_wrong() const { return max_wrong; }

std::string Game::reveal_solution() const {
    return candidates.empty() ? std::string() : candidates.front()->text;
}
