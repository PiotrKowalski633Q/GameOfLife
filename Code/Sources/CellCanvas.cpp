#include "../Headers/CellCanvas.h"

#define spriteCanvasToScreenProportion 0.85f

CellCanvas::CellCanvas(int screenWidth, int screenHeight, int columnCount, int rowCount)
:mScreenWidth(screenWidth),
mScreenHeight(screenHeight),
mColumnCount(columnCount),
mRowCount(rowCount),
mTimeSinceLastUpdate(0),
mUpdateInterval(1000000),
mUpdateIntervalDivider(1)
{
    mDeadCellTexture.loadFromFile("Resources/Images/deadCell.png");
    mAliveCellTexture.loadFromFile("Resources/Images/aliveCell.png");

    updateCellsAndSpritesToMatchColumnsAndRows();

    //sets default platform and device to use for kernel calculations, allDevicesOnDefaultPlatform[0] is CPU, allDevicesOnDefaultPlatform[1] is GPU
    std::vector<cl::Platform> allPlatforms = OpenCLFunctions::getAllPlatforms();
    mOpenCLObject.platform = allPlatforms[0];
    std::vector<cl::Device> allDevicesOnDefaultPlatform = OpenCLFunctions::getAllDevicesOnPlatform(mOpenCLObject.platform);
    //GPU should generally be used whenever available, but not every PC has a separate GPU
    if (allDevicesOnDefaultPlatform.size() < 2)
    {
        mOpenCLObject.device = allDevicesOnDefaultPlatform[0];
    }
    else
    {
        mOpenCLObject.device = allDevicesOnDefaultPlatform[1];
    }

    //sets OpenCL context and begins allocating memory on the device using OpenCL buffer objects
    mOpenCLObject.context = cl::Context({mOpenCLObject.device});
    OpenCLFunctions::allocateMemoryOnDevice(mOpenCLObject.deviceColumnCount, 1*sizeof(int), mOpenCLObject.context);
    OpenCLFunctions::allocateMemoryOnDevice(mOpenCLObject.deviceRowCount, 1*sizeof(int), mOpenCLObject.context);
    
    OpenCLFunctions::allocateMemoryOnDevice(mOpenCLObject.deviceInputCellValues, mColumnCount*mRowCount*sizeof(int), mOpenCLObject.context);
    OpenCLFunctions::allocateMemoryOnDevice(mOpenCLObject.deviceOutputCellValues, mColumnCount*mRowCount*sizeof(int), mOpenCLObject.context);

    //sets remaining OpenCL objects including two kernels with two separate programs that will be used during fractal generation
    mOpenCLObject.programCell = OpenCLFunctions::buildProgramFromFile(mOpenCLObject.device, mOpenCLObject.context, "Resources/Kernels/cell.txt");
    mOpenCLObject.commandQueue = cl::CommandQueue(mOpenCLObject.context, mOpenCLObject.device);
    mOpenCLObject.bufferOutputCellValues = new int[mColumnCount*mRowCount*3];
    mOpenCLObject.kernelCell = OpenCLFunctions::createKernelForProgram("cell", mOpenCLObject.programCell, {mOpenCLObject.deviceColumnCount, mOpenCLObject.deviceRowCount, mOpenCLObject.deviceInputCellValues, mOpenCLObject.deviceOutputCellValues});

    //local and global work sizes can only be decided after kernels are created
    int bestLocalWorkgroupSizePerDimension = OpenCLFunctions::findBestLocalWorkgroupSizePerDimension(mOpenCLObject.kernelCell, mOpenCLObject.device);
    mOpenCLObject.localWorkGroupSize = cl::NDRange(bestLocalWorkgroupSizePerDimension, bestLocalWorkgroupSizePerDimension);
    mOpenCLObject.globalWorkGroupSize = OpenCLFunctions::findBestGlobalWorkgroupSize(bestLocalWorkgroupSizePerDimension, mColumnCount, mRowCount);

    //finally send to the device all the data that is already known
    int arrayFormColumnCount[1] = {mColumnCount};
    int arrayFormRowCount[1] = {mRowCount};
    OpenCLFunctions::sendDataToDevice((void*)arrayFormColumnCount, mOpenCLObject.deviceColumnCount, 1*sizeof(int), mOpenCLObject.commandQueue);
    OpenCLFunctions::sendDataToDevice((void*)arrayFormRowCount, mOpenCLObject.deviceRowCount, 1*sizeof(int), mOpenCLObject.commandQueue);
}

CellCanvas::~CellCanvas()
{

}

TwoValueKey CellCanvas::getCellByPositionOnScreen(sf::Vector2<int> position)
{
    if (mMapOfSprites.find(TwoValueKey(0,0)) == mMapOfSprites.end())
    {
        return TwoValueKey(-1,-1);
    }

    float spriteScale = mMapOfSprites.find(TwoValueKey(0,0))->second.getScale().x;
    float spriteSize = spriteScale*100;
    for (std::map<TwoValueKey, sf::Sprite>::iterator iter=mMapOfSprites.begin(); iter != mMapOfSprites.end(); iter++)
    {
        sf::Vector2 spritePosition = iter->second.getPosition();
        if (position.x>=spritePosition.x-(spriteSize/2) && position.x<=spritePosition.x+(spriteSize/2))
        {
            if (position.y>=spritePosition.y-(spriteSize/2) && position.y<=spritePosition.y+(spriteSize/2))
            {
                return iter->first;
            }
        }
    }

    return TwoValueKey(-1,-1);
}

void CellCanvas::switchCellState(TwoValueKey cell)
{
    std::map<TwoValueKey, int>::iterator iterCell = mMapOfCells.find(cell);
    if (iterCell == mMapOfCells.end())
    {
        return;
    }

    if (iterCell->second == 1)
    {
        iterCell->second = 0;
    }
    else
    {
        iterCell->second = 1;
    }

    std::map<TwoValueKey, sf::Sprite>::iterator iterSprite = mMapOfSprites.find(cell);
    if (iterCell->second == 1)
    {
        iterSprite->second.setTexture(mAliveCellTexture);
    }
    else
    {
        iterSprite->second.setTexture(mDeadCellTexture);
    }
}

void CellCanvas::addColumn()
{
    mColumnCount += 1;
    updateCellsAndSpritesToMatchColumnsAndRows();
    updateOpenCLObjectToMatchColumnsAndRows();
}

void CellCanvas::addRow()
{
    mRowCount += 1;
    updateCellsAndSpritesToMatchColumnsAndRows();
    updateOpenCLObjectToMatchColumnsAndRows();
}

void CellCanvas::removeColumn()
{
    if (mColumnCount > 1)
    {
        mColumnCount -= 1;
        updateCellsAndSpritesToMatchColumnsAndRows();
        updateOpenCLObjectToMatchColumnsAndRows();
    }
}

void CellCanvas::removeRow()
{
    if (mRowCount > 1)
    {
        mRowCount -= 1;
        updateCellsAndSpritesToMatchColumnsAndRows();
        updateOpenCLObjectToMatchColumnsAndRows();
    }
}

void CellCanvas::speedUpUpdateInterval()
{
    mUpdateIntervalDivider++;
}

void CellCanvas::slowDownUpdateInterval()
{
    if (mUpdateIntervalDivider == 1)
    {
        return;
    }
    mUpdateIntervalDivider--;
}

void CellCanvas::update(double deltaTime)
{
    mTimeSinceLastUpdate += deltaTime;
    if (mTimeSinceLastUpdate >= (mUpdateInterval/mUpdateIntervalDivider))
    {
        mTimeSinceLastUpdate -= (mUpdateInterval/mUpdateIntervalDivider);
        updateCells();
        updateSpritesToMatchCellStates();
    }
}

void CellCanvas::draw(sf::RenderWindow &window)
{
    for (std::map<TwoValueKey, sf::Sprite>::iterator iter=mMapOfSprites.begin(); iter != mMapOfSprites.end(); iter++)
    {
        window.draw(iter->second);
    }
}

void CellCanvas::updateCells()
{
    //OpenCL only likes arrays, so convert map to array
    int arrayFormInputCellValues[mColumnCount*mRowCount];
    for (std::map<TwoValueKey, int>::iterator iterCell=mMapOfCells.begin(); iterCell != mMapOfCells.end(); iterCell++)
    {
        arrayFormInputCellValues[iterCell->first.x*mRowCount+iterCell->first.y] = iterCell->second;
    }

    //sends all the necessary (changing between frames) data to kernels
    OpenCLFunctions::sendDataToDevice((void*)arrayFormInputCellValues, mOpenCLObject.deviceInputCellValues, mColumnCount*mRowCount*sizeof(int), mOpenCLObject.commandQueue);

    //begins calculating new cell values for every cell
    OpenCLFunctions::startKernel(mOpenCLObject.kernelCell, mOpenCLObject.commandQueue, mOpenCLObject.localWorkGroupSize, mOpenCLObject.globalWorkGroupSize);

    //waits for kernels to finish all their actions...
    mOpenCLObject.commandQueue.finish();

    //...before retrieving results...
    OpenCLFunctions::getDataFromDevice((void*)mOpenCLObject.bufferOutputCellValues, mOpenCLObject.deviceOutputCellValues, mColumnCount*mRowCount*sizeof(int), mOpenCLObject.commandQueue);

    //...and using them to set cell values within canvas
    for (int i=0; i<mColumnCount; i++)
    {
        for (int j=0; j<mRowCount; j++)
        {
            mMapOfCells.find(TwoValueKey(i,j))->second = mOpenCLObject.bufferOutputCellValues[i*mRowCount+j];
        }
    }
}

void CellCanvas::updateSpritesToMatchCellStates()
{
    for (std::map<TwoValueKey, int>::iterator iterCell=mMapOfCells.begin(); iterCell != mMapOfCells.end(); iterCell++)
    {
        std::map<TwoValueKey, sf::Sprite>::iterator iterSprite=mMapOfSprites.find(iterCell->first);
        if (iterCell->second == 1)
        {
            iterSprite->second.setTexture(mAliveCellTexture);
        }
        else
        {
            iterSprite->second.setTexture(mDeadCellTexture);
        }
    }
}

void CellCanvas::updateCellsAndSpritesToMatchColumnsAndRows()
{
    float widthBasedSpriteScale = ((float)mScreenWidth/mColumnCount)/100*spriteCanvasToScreenProportion;
    float heightBasedSpriteScale = ((float)mScreenHeight/mRowCount)/100*spriteCanvasToScreenProportion;
    float spriteScale = std::min(widthBasedSpriteScale, heightBasedSpriteScale);
    mMapOfCells.clear();
    mMapOfSprites.clear();
    for (int i=0; i<mColumnCount; i++)
    {
        for (int j=0; j<mRowCount; j++)
        {
            mMapOfCells.insert(std::make_pair(TwoValueKey(i, j), 0));
            sf::Sprite sprite;
            sprite.setTexture(mDeadCellTexture);
            sprite.setScale(spriteScale, spriteScale);
            sprite.setOrigin(50, 50);
            sprite.setPosition(mScreenWidth/2.0+spriteScale*100*(i-mColumnCount/2.0+0.5), mScreenHeight/2.0+spriteScale*100*(j-mRowCount/2.0+0.5));
            mMapOfSprites.insert(std::make_pair(TwoValueKey(i, j), sprite));
        }
    }
}

void CellCanvas::updateOpenCLObjectToMatchColumnsAndRows()
{
    mOpenCLObject.deviceInputCellValues = cl::Buffer();
    mOpenCLObject.deviceOutputCellValues = cl::Buffer();
    OpenCLFunctions::allocateMemoryOnDevice(mOpenCLObject.deviceInputCellValues, mColumnCount*mRowCount*sizeof(int), mOpenCLObject.context);
    OpenCLFunctions::allocateMemoryOnDevice(mOpenCLObject.deviceOutputCellValues, mColumnCount*mRowCount*sizeof(int), mOpenCLObject.context);
    mOpenCLObject.kernelCell = OpenCLFunctions::createKernelForProgram("cell", mOpenCLObject.programCell, {mOpenCLObject.deviceColumnCount, mOpenCLObject.deviceRowCount, mOpenCLObject.deviceInputCellValues, mOpenCLObject.deviceOutputCellValues});
    int arrayFormColumnCount[1] = {mColumnCount};
    int arrayFormRowCount[1] = {mRowCount};
    OpenCLFunctions::sendDataToDevice((void*)arrayFormColumnCount, mOpenCLObject.deviceColumnCount, 1*sizeof(int), mOpenCLObject.commandQueue);
    OpenCLFunctions::sendDataToDevice((void*)arrayFormRowCount, mOpenCLObject.deviceRowCount, 1*sizeof(int), mOpenCLObject.commandQueue);
    int bestLocalWorkgroupSizePerDimension = OpenCLFunctions::findBestLocalWorkgroupSizePerDimension(mOpenCLObject.kernelCell, mOpenCLObject.device);
    mOpenCLObject.localWorkGroupSize = cl::NDRange(bestLocalWorkgroupSizePerDimension, bestLocalWorkgroupSizePerDimension);
    mOpenCLObject.globalWorkGroupSize = OpenCLFunctions::findBestGlobalWorkgroupSize(bestLocalWorkgroupSizePerDimension, mColumnCount, mRowCount);
}
