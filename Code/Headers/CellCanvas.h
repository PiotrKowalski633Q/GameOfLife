#ifndef GAMEOFLIFE_CELLCANVAS
#define GAMEOFLIFE_CELLCANVAS

#include <SFML/Graphics.hpp>
#include <CL/cl.hpp>

#include "OpenCLFunctions.h"

struct TwoValueKey
{
    int x, y;
    bool operator< (const TwoValueKey other) const
    {
        if (this->x < other.x)
        {
            return true;
        }
        else if (this->x == other.x)
        {
            if (this->y < other.y)
            {
                return true;
            }
        }
        return false;
    }

    TwoValueKey(int x, int y)
    {
        this->x = x;
        this->y = y;
    }
};

struct OpenCLObject
{
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program programCell;
    cl::CommandQueue commandQueue;
    cl::Kernel kernelCell;
    cl::Buffer deviceColumnCount, deviceRowCount, deviceInputCellValues, deviceOutputCellValues;

    cl::NDRange localWorkGroupSize, globalWorkGroupSize;

    int* bufferOutputCellValues;
};

class CellCanvas
{
private:
    int mScreenWidth, mScreenHeight;
    int mColumnCount, mRowCount;
    double mTimeSinceLastUpdate;
    double mUpdateInterval;
    int mUpdateIntervalDivider;

    sf::Texture mDeadCellTexture;
    sf::Texture mAliveCellTexture;

    std::map<TwoValueKey, int> mMapOfCells;
    std::map<TwoValueKey, sf::Sprite> mMapOfSprites;

    OpenCLObject mOpenCLObject;

public:
    CellCanvas(int screenWidth, int screenHeight, int columnCount, int rowCount);

    ~CellCanvas();

    TwoValueKey getCellByPositionOnScreen(sf::Vector2<int> position);

    void switchCellState(TwoValueKey cell);
    void addColumn();
    void addRow();
    void removeColumn();
    void removeRow();
    void speedUpUpdateInterval();
    void slowDownUpdateInterval();

    void update(double deltaTime);

    void draw(sf::RenderWindow &window);

private:
    void updateCells();
    void updateSpritesToMatchCellStates();
    void updateCellsAndSpritesToMatchColumnsAndRows();
    void updateOpenCLObjectToMatchColumnsAndRows();
};

#endif //GAMEOFLIFE_CELLCANVAS