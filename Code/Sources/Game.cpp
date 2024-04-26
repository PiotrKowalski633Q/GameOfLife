#include "../Headers/Game.h"

Game::Game()
:mWindow(sf::RenderWindow( sf::VideoMode( GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), 32 ), "Game Of Life", sf::Style::Fullscreen )),
mDeltaTime(0),
mRenderingFrameTimer(0),
mExpectedRenderingFps(30),
mCellCanvas(GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), 30, 20),
mIsPaused(false)
{
    mBackgroundTexture.setRepeated(true);
    mBackgroundTexture.loadFromFile("Resources/Images/background.png");
    mBackgroundSprite.setTexture(mBackgroundTexture);
}

void Game::gameLoop()//main loop, it will continuously poll events, read them and terminate only if the window closes
{
    mFrameBegin = std::chrono::steady_clock::now();

    while (mWindow.isOpen())
    {
        mFrameEnd = std::chrono::steady_clock::now();
        mDeltaTime = std::chrono::duration_cast<std::chrono::microseconds>(mFrameEnd - mFrameBegin).count();
        mRenderingFrameTimer += mDeltaTime;
        mFrameBegin = std::chrono::steady_clock::now();

        processInput();

        if (!mIsPaused)
        {
            update();
        }

        if (1000000.0/mRenderingFrameTimer <= mExpectedRenderingFps)
        {
            draw();
            mRenderingFrameTimer -= 1000000.0/mExpectedRenderingFps;
        }
        else
        {
            //this way the loop will always advance at least 1/10 of the way towards next draw call when not drawing and prevent extremely low deltaTime from causing precision problems
            std::this_thread::sleep_for(std::chrono::microseconds((int)(1000000/mExpectedRenderingFps/10)));
        }
    }
}

void Game::processInput()
{
    sf::Event event;
    while (mWindow.pollEvent( event ))
    {
        if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)
        {
            mWindow.close();
            break;
        }

        if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::LShift)
        {
            mCellCanvas.speedUpUpdateInterval();
        }
        else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::LAlt)
        {
            mCellCanvas.slowDownUpdateInterval();
        }

        if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Right)
        {
            mCellCanvas.addColumn();
        }
        else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Down)
        {
            mCellCanvas.addRow();
        }
        else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Left)
        {
            mCellCanvas.removeColumn();
        }
        else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Up)
        {
            mCellCanvas.removeRow();
        }

        if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Space)
        {
            if (mIsPaused)
            {
                mIsPaused = false;
            }
            else
            {
                mIsPaused = true;
            }
        }

        if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Button::Left)
        {
            mCellCanvas.switchCellState(mCellCanvas.getCellByPositionOnScreen(sf::Mouse::getPosition(mWindow)));
        }
    }
}

void Game::update()
{
    mCellCanvas.update(mDeltaTime);
}

void Game::draw()
{
    mWindow.clear();
    if (mIsPaused)
    {
        mBackgroundSprite.setColor(sf::Color(255, 200, 200));
    }
    else
    {
        mBackgroundSprite.setColor(sf::Color(255, 255, 255));
    }
    mWindow.draw(mBackgroundSprite);
    mCellCanvas.draw(mWindow);
    mWindow.display();
}

void Game::run()
{

    sf::Music music;
    music.openFromFile("Resources/Music/february-night.mp3");
    music.setLoop(true);
    music.play();

    gameLoop();
}