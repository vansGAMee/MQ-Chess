// libanalyzer.cpp – Loss Index (абсолютная метрика)
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdio>

extern "C" {

struct MoveInput {
    int bestEval;
    int playedEval;
    int legalMoves;
    int secondBestEval;
    double timeSpent;
    bool isMate;
};

inline double cpToWinChance(int cp) {
    if (cp > 1000) return 1.0;
    if (cp < -1000) return -1.0;
    return 2.0 / (1.0 + std::exp(-0.003624 * cp)) - 1.0;
}

double calculate_advanced_mq(MoveInput* moves, int count) {
    if (!moves || count == 0) return -1.0;   // ошибка

    double sumDeltaWeighted = 0.0;
    double sumWeights = 0.0;
    int totalPassed = 0;

    for (int i = 0; i < count; ++i) {
        MoveInput m = moves[i];

        if (m.legalMoves == -1) continue;  // разделители не учитываем

        // Фильтры
        if (m.legalMoves <= 1) continue;
        if (m.timeSpent >= 0.0 && m.timeSpent < 0.4) continue;

        double Wbest = cpToWinChance(m.bestEval);
        double Wplayed = cpToWinChance(m.playedEval);
        double deltaW = std::abs(Wbest - Wplayed);
        double weight = std::log(m.legalMoves + 1.0);

        sumDeltaWeighted += deltaW * weight;
        sumWeights += weight;
        totalPassed++;
    }

    if (sumWeights == 0.0 || totalPassed == 0) return -1.0;

    double avgDeltaW = sumDeltaWeighted / sumWeights;
    double lossIndex = avgDeltaW * 100.0;   // в процентах

    printf("[C++ DEBUG] Ходов обработано: %d\n", totalPassed);
    printf("[C++ DEBUG] Loss Index: %.2f%%\n", lossIndex);
    fflush(stdout);

    return lossIndex;
}

} // extern "C"