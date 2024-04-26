#ifndef GAMEOFLIFE_GAME
#define GAMEOFLIFE_GAME

#include <chrono>
#include <wtypes.h>
#include <thread>

#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>

#include "CellCanvas.h"

class Game
{
    sf::RenderWindow mWindow;

    std::chrono::steady_clock::time_point mFrameBegin;
    std::chrono::steady_clock::time_point mFrameEnd;
    double mDeltaTime;
    double mRenderingFrameTimer;
    double mExpectedRenderingFps;

    CellCanvas mCellCanvas;

    sf::Texture mBackgroundTexture;
    sf::Sprite mBackgroundSprite;

    bool mIsPaused;

public:
    Game();

private:
    void gameLoop();

    void processInput();
    void update();
    void draw();

public:
    void run();
};

#endif //GAMEOFLIFE_GAME
