#include "mainwindow.h"
#include "logger.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QtWidgets>
#include <array>

#if defined(QT_MULTIMEDIA_LIB)
#include <QtMultimedia>
#endif

std::vector<StipplerLayer> toStipplerLayers(const Layers& layers) {
    std::vector<StipplerLayer> stipplerLayers;
    for (const auto& layer : layers) {
        stipplerLayers.push_back(layer.stippler);
    }
    return std::move(stipplerLayers);
};

MainWindow::MainWindow(Layers layers, QWidget* parent)
    : QMainWindow(parent)
    , m_layers() {
    m_imageViewer = new ImageViewer(this);
    setCentralWidget(m_imageViewer);

    QDockWidget* toolboxDock = new QDockWidget("Toolbox", this);
    toolboxDock->setFeatures(QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetMovable);
    toolboxDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    addDockWidget(Qt::RightDockWidgetArea, toolboxDock);

    m_toolbox = new ToolboxWidget(m_imageViewer->canvasColor(), m_layers, m_stippler, toolboxDock);
    connect(m_toolbox, &ToolboxWidget::start, [this]() { this->stipple(); });
    connect(m_toolbox, &ToolboxWidget::layerStyleChanged, this, &MainWindow::updateImageViewer);
    connect(m_toolbox, &ToolboxWidget::minIterationDurationChanged, [this](int durationMs) {
        setIntermediateResultDisplay(durationMs);
    });
    connect(m_toolbox, &ToolboxWidget::canvasColorChanged, [this](QColor canvasColor) {
        m_imageViewer->setCanvasColor(canvasColor);
    });
    connect(m_toolbox, &ToolboxWidget::renderModeChanged, [this](RenderMode mode) {
        this->updateImageViewer();
    });
    connect(m_toolbox, &ToolboxWidget::importAsDensity, this, &MainWindow::loadImages);
    connect(m_toolbox, &ToolboxWidget::importAsDual, this, &MainWindow::loadImageAsDual);
#if defined(QT_MULTIMEDIA_LIB)
    connect(m_toolbox, &ToolboxWidget::importCamera, this, &MainWindow::loadCamera);
#endif
    connect(m_toolbox, &ToolboxWidget::importLayers, this, &MainWindow::loadLayers);
    connect(m_toolbox, &ToolboxWidget::exportImage, this, &MainWindow::saveImage);
    connect(m_toolbox, &ToolboxWidget::exportNaturalNeighborData, this, &MainWindow::computeAndSaveNaturalNeighborData);
    connect(m_toolbox, &ToolboxWidget::loadProject, this, &MainWindow::loadProject);
    connect(m_toolbox, &ToolboxWidget::saveProject, this, &MainWindow::saveProject);
#ifdef _DEBUG
    m_toolbox->setMinIterationDuration(125);
#endif
    toolboxDock->setWidget(m_toolbox);

    m_imageViewer->setHightlightSplits(true);
    setLayers(std::move(layers));
}

QJsonObject MainWindow::parametersToJson() {
    QJsonObject json;
    // Common parameters.
    const auto& common = m_stippler.options();
    QJsonObject jsonCommon;
    jsonCommon.insert("stippleAlgorithm", QJsonValue::fromVariant(QVariant::fromValue(common.stippleAlgorithm)));
    jsonCommon.insert("voronoiAlgorithm", QJsonValue::fromVariant(QVariant::fromValue(common.voronoiAlgorithm)));
    jsonCommon.insert("voronoiScale", QJsonValue(common.voronoiScale));
    jsonCommon.insert("borderWidth", QJsonValue(common.borderWidth));
    QJsonObject jsonCommonLbg;
    jsonCommonLbg.insert("stippleModel", QJsonValue::fromVariant(QVariant::fromValue(common.lbg.stippleModel)));
    jsonCommonLbg.insert("maxIterations", QJsonValue(common.lbg.maxIterations));
    jsonCommonLbg.insert("hysteresisStart", QJsonValue(common.lbg.hysteresisStart));
    jsonCommonLbg.insert("hysteresisMax", QJsonValue(common.lbg.hysteresisMax));
    jsonCommonLbg.insert("hysteresisFunction", QJsonValue::fromVariant(QVariant::fromValue(common.lbg.hysteresisFunction)));
    jsonCommonLbg.insert("forcedCooldown", QJsonValue(common.lbg.forcedCooldown));
    jsonCommonLbg.insert("splitJitterSpread", QJsonValue(common.lbg.splitJitterSpread));
    jsonCommonLbg.insert("errorReductionChance", QJsonValue(common.lbg.errorReductionChance));
    jsonCommon.insert("lbg", jsonCommonLbg);
    json.insert("common", jsonCommon);
    // Layer-specific parameters.
    QJsonArray jsonLayers;
    for (const auto& layer : m_layers) {
        QJsonObject jsonLayer;
        jsonLayer.insert("imagePath", QJsonValue(layer.imagePath));
        jsonLayer.insert("initialStipples", QJsonValue(layer.stippler.initialStipples));
        QJsonObject jsonLayerLbg;
        QColor qColor(layer.stippler.lbg.color.r(), layer.stippler.lbg.color.g(), layer.stippler.lbg.color.b(), layer.stippler.lbg.color.a());
        jsonLayerLbg.insert("color", QJsonValue::fromVariant(qColor));
        jsonLayerLbg.insert("shape", QJsonValue::fromVariant(QVariant::fromValue(layer.stippler.lbg.shape)));
        jsonLayerLbg.insert("shapeParameter", QJsonValue(layer.stippler.lbg.shapeParameter));
        jsonLayerLbg.insert("shapeRadius", QJsonValue(layer.stippler.lbg.shapeRadius));
        jsonLayerLbg.insert("sizeMin", QJsonValue(layer.stippler.lbg.sizeMin));
        jsonLayerLbg.insert("sizeMax", QJsonValue(layer.stippler.lbg.sizeMax));
        jsonLayerLbg.insert("sizeFunction", QJsonValue::fromVariant(QVariant::fromValue(layer.stippler.lbg.sizeFunction)));
        jsonLayer.insert("lbg", jsonLayerLbg);
        jsonLayers.append(jsonLayer);
    }
    json.insert("layers", jsonLayers);
    return json;
}

void MainWindow::parametersFromJson(QJsonObject json, const QString* path) {
    bool ok = true;
    QFileInfo pathInfo(path ? *path : "");
    QJsonObject jsonCommon = json["common"].toObject();
    StipplerOptions common;
    common.stippleAlgorithm = jsonCommon["stippleAlgorithm"].toVariant().value<StippleAlgorithm>();
    common.voronoiAlgorithm = jsonCommon["voronoiAlgorithm"].toVariant().value<VoronoiAlgorithm>();
    common.voronoiScale = jsonCommon["voronoiScale"].toDouble(common.voronoiScale);
    common.borderWidth = jsonCommon["borderWidth"].toInt(common.borderWidth);
    QJsonObject jsonCommonLbg = jsonCommon["lbg"].toObject();
    common.lbg.stippleModel = jsonCommonLbg["stippleModel"].toVariant().value<StippleModel>();
    common.lbg.maxIterations = jsonCommonLbg["maxIterations"].toInt(common.lbg.maxIterations);
    common.lbg.hysteresisStart = jsonCommonLbg["hysteresisStart"].toDouble(common.lbg.hysteresisStart);
    common.lbg.hysteresisMax = jsonCommonLbg["hysteresisMax"].toDouble(common.lbg.hysteresisMax);
    common.lbg.hysteresisFunction = jsonCommonLbg["hysteresisFunction"].toVariant().value<HysteresisFunction>();
    common.lbg.forcedCooldown = jsonCommonLbg["forcedCooldown"].toBool(common.lbg.forcedCooldown);
    common.lbg.splitJitterSpread = jsonCommonLbg["splitJitterSpread"].toDouble(common.lbg.splitJitterSpread);
    common.lbg.errorReductionChance = jsonCommonLbg["errorReductionChance"].toDouble(common.lbg.errorReductionChance);
    setOptions(std::move(common));
    // Layer-specific parameters.
    Layers layers;
    QJsonArray jsonLayers = json["layers"].toArray();
    for (const auto& jsonLayer : jsonLayers) {
        QJsonObject jsonLayerObject = jsonLayer.toObject();
        Layer layer;
        if (pathInfo.exists()) {
            layer.imagePath = pathInfo.dir().filePath(jsonLayerObject["imagePath"].toString());
        } else {
            layer.imagePath = jsonLayerObject["imagePath"].toString();
        }
        layer.image = QImage(layer.imagePath);
        if (layer.image.isNull()) {
            ok = false;
        }
        layer.stippler.initialStipples = jsonLayerObject["initialStipples"].toInt(layer.stippler.initialStipples);
        QJsonObject jsonLayerLbg = jsonLayerObject["lbg"].toObject();
        layer.setColor(jsonLayerLbg["color"].toVariant().value<QColor>());
        layer.stippler.lbg.shape = jsonLayerLbg["shape"].toVariant().value<StippleShape>();
        layer.stippler.lbg.shapeParameter = jsonLayerLbg["shapeParameter"].toDouble(layer.stippler.lbg.shapeParameter);
        layer.stippler.lbg.shapeRadius = jsonLayerLbg["shapeRadius"].toDouble(layer.stippler.lbg.shapeRadius);
        layer.stippler.lbg.sizeMin = jsonLayerLbg["sizeMin"].toDouble(layer.stippler.lbg.sizeMin);
        layer.stippler.lbg.sizeMax = jsonLayerLbg["sizeMax"].toDouble(layer.stippler.lbg.sizeMax);
        layer.stippler.lbg.sizeFunction = jsonLayerLbg["sizeFunction"].toVariant().value<SizeFunction>();
        layers.append(std::move(layer));
    }
    if (ok) {
        setLayers(std::move(layers));
    } else {
        qWarning() << "Failed loading project";
    }
}

void MainWindow::setCanvasColor(QColor color) {
    m_toolbox->setCanvasColor(color);
    m_imageViewer->setCanvasColor(color);
}

void MainWindow::setIntermediateResultDisplay(int minIterationDuration) {
    if (minIterationDuration > 0) {
        setIterationCallback([this, minIterationDuration](const auto& layerStipples, const auto& lindeBuzoGrayResult) {
            for (int layerIndex = 0; layerIndex < m_layers.size(); layerIndex++) {
                m_layers[layerIndex].stipples = layerStipples[layerIndex];
            }
            if (!lindeBuzoGrayResult.done || m_frameChanging) {
                m_toolbox->setRenderMode(RenderMode::RasterStipples);
            } else {
                m_toolbox->setRenderMode(RenderMode::RasterStipplesWithBackground);
            }
            updateImageViewer();
            QCoreApplication::processEvents();

            quint64 waitTime = qMax<qint64>(minIterationDuration - m_stippleTimer.restart(), 0);
            QThread::msleep(waitTime);
        });
    } else {
        setIterationCallback(nullptr);
    }
}

void MainWindow::setIterationCallback(Stippler::IterationCallback cb) {
    m_stippler.setIterationCallback(cb);
}

void MainWindow::setOptions(StipplerOptions options) {
    m_stippler.setOptions(std::move(options));
    m_toolbox->invalidateParameterWidgets();
}

void MainWindow::setLayers(Layers layers) {
    m_layers = std::move(layers);
    for (auto& layer : m_layers) {
        std::vector<float> pixels;
        pixels.reserve(layer.image.width() * layer.image.height());
        for (int y = 0; y < layer.image.height(); ++y) {
            for (int x = 0; x < layer.image.width(); ++x) {
                QRgb pixel = layer.image.pixel(x, y);
                float density = (1.0f - qGray(pixel) / 255.0f) * qAlpha(pixel) / 255.0f;
                density = std::max(0.0f, std::min(1.0f, density));
                pixels.push_back(density);
            }
        }
        layer.stippler.density = Map<float> {
            layer.image.width(),
            layer.image.height(),
            std::move(pixels)
        };
    }

    m_toolbox->invalidateLayerWidgets();
    m_toolbox->setRenderMode(RenderMode::PainterDensity);
    updateImageViewer();
    m_imageViewer->fitInView(m_imageViewer->scene()->sceneRect(), Qt::KeepAspectRatio);
}

void MainWindow::loadImages(const QStringList& paths) {
    std::array<QColor, 20> defaultColors = {
        QColor(0xFFB300), QColor(0x803E75), QColor(0xFF6800), QColor(0xA6BDD7), QColor(0xC10020),
        QColor(0xCEA262), QColor(0x817066), QColor(0x007D34), QColor(0xF6768E), QColor(0x00538A),
        QColor(0xFF7A5C), QColor(0x53377A), QColor(0xFF8E00), QColor(0xB32851), QColor(0xF4C800),
        QColor(0x7F180D), QColor(0x93AA00), QColor(0x593315), QColor(0xF13A13), QColor(0x232C16)
    };
    QRegularExpression reColor("_(?<COLOR>[0-9A-Fa-f]{6})");
    Layers layers;
    for (int i = 0; i < paths.size(); ++i) {
        auto path = paths[i];
        Layer layer(QImage(path), path, Qt::black);
        QString colorString = reColor.match(path).captured("COLOR");
        if (!colorString.isNull()) {
            layer.setColor(QColor("#" + colorString));
        } else {
            if (layers.isEmpty() && paths.size() == 1) {
                layer.setColor(Qt::black);
                setCanvasColor(Qt::white);
            } else {
                layer.setColor(defaultColors[i % defaultColors.size()]);
            }
        }
        if (layers.isEmpty() || (!layers.isEmpty() && layers.front().image.size() == layer.image.size())) {
            layers.append(layer);
        } else {
            qWarning() << "Images must have same size";
        }
    }
    setLayers(std::move(layers));
}

void MainWindow::loadImageAsDual(const QString& path) {
    QImage image = QImage(path).convertToFormat(QImage::Format_Grayscale8);
    QImage imageDual(image);
    imageDual.invertPixels(QImage::InvertRgb);
    Layers layers;
    Layer blackLayer(image, path, Qt::black);
    Layer whiteLayer(imageDual, path, Qt::white);
    layers.append(blackLayer);
    layers.append(whiteLayer);
    setLayers(std::move(layers));
}

void MainWindow::loadLayers(Layers layers) {
    setLayers(std::move(layers));
}

#if defined(QT_MULTIMEDIA_LIB)
void MainWindow::loadCamera(QCamera* camera) {
    // TODO: delete old camera/worker/etc.

    QThread* thread = new QThread();
    FrameWorker* frameWorker = new FrameWorker(m_frameChanging);
    connect(this, &MainWindow::cameraFrameChanged, frameWorker, &FrameWorker::processVideoFrame);
    connect(frameWorker, &FrameWorker::mapsChanged, this, &MainWindow::loadCameraMaps);
    frameWorker->moveToThread(thread);
    thread->start();

    QMediaCaptureSession* captureSession = new QMediaCaptureSession();
    QVideoSink* videoSink = new QVideoSink();
    connect(videoSink, QOverload<const QVideoFrame&>::of(&QVideoSink::videoFrameChanged), [this, frameWorker](const QVideoFrame& frame) {
        if (!frame.isValid()) {
            return;
        }
        if (frameWorker->isReady()) {
            emit cameraFrameChanged(frame);
        }
    });
    captureSession->setVideoSink(videoSink);
    captureSession->setCamera(camera);

    // m_toolbox->setmax(RenderMode::PainterHighlightedStipples);
    m_layers[0].image = QImage(videoSink->videoSize() / 2, QImage::Format_ARGB32); // XXX: another hack.
    m_toolbox->setRenderMode(RenderMode::RasterStipples);
    m_toolbox->setMinIterationDuration(1);
    m_toolbox->setEditable(false);
    auto docks = findChildren<QDockWidget*>();
    for (auto* dock : docks) {
        dock->hide();
    }
    QCoreApplication::processEvents();

    camera->start();
}

void MainWindow::loadCameraMaps(std::vector<Map<float>> maps) {
    if (m_frameChanging) {
        return;
    }
    m_frameChanging = true;

    // qDebug() << "loadCameraMaps";

    auto stipplerLayers = toStipplerLayers(m_layers);
    for (int i = 0; i < stipplerLayers.size(); ++i) {
        stipplerLayers[i].density = maps[i];
    }
    m_stippler.resetLayers(stipplerLayers, true);
    m_stippler.stipple();

    m_frameChanging = false;
}
#endif

void MainWindow::loadProject(const QString& path) {
    QFile jsonFile(path);
    jsonFile.open(QFile::ReadOnly);
    QJsonDocument jsonDoc = QJsonDocument::fromJson(jsonFile.readAll());
    jsonFile.close();
    parametersFromJson(std::move(jsonDoc.object()), &path);
}

void MainWindow::saveProject(const QString& path) {
    QJsonDocument jsonDoc(parametersToJson());
    QFile jsonFile(path);
    jsonFile.open(QFile::WriteOnly);
    jsonFile.write(jsonDoc.toJson());
    jsonFile.close();
}

void MainWindow::saveImage(const QString& path) {
    m_imageViewer->save(path);
}

qint64 MainWindow::stipple() {
    if (m_toolbox->renderMode() == RenderMode::PainterDensity) {
#if _DEBUG
        m_toolbox->setRenderMode(RenderMode::PainterHighlightedStipples);
#else
        m_toolbox->setRenderMode(RenderMode::RasterStipplesWithBackground);
#endif
    }
    m_toolbox->invalidateParameterWidgets();
    m_toolbox->setEditable(false);
    QCoreApplication::processEvents();

    m_stippleTimer.start();
    m_stippler.resetLayers(toStipplerLayers(m_layers));

    QElapsedTimer executionTimer;
    executionTimer.start();
    auto layerStipples = m_stippler.stipple();
    qint64 executionDuration = executionTimer.elapsed();

    m_stippleTimer.invalidate();

    for (int layerIndex = 0; layerIndex < m_layers.size(); layerIndex++) {
        m_layers[layerIndex].stipples = layerStipples[layerIndex];
    }
    updateImageViewer();
    m_toolbox->setEditable(true);

    return executionDuration;
}

void MainWindow::computeAndSaveNaturalNeighborData() {
    qInfo() << "Computing natural neighbor maps...";

    // TODO: evil constants.
    const size_t BucketSize = 16;
    const int KernelSize = 128;

    auto naturalNeighborData = m_stippler.computeNaturalNeighbors(BucketSize, KernelSize);
    if (!naturalNeighborData) {
        qCritical() << "Failed (bucket size too small)";
        return;
    }

    qInfo() << "Writing files...";

    m_imageViewer->save("nn.positions.image.png");

    QFile positionsFile("nn.positions.i32.bin");
    positionsFile.open(QIODevice::WriteOnly);
    positionsFile.write(reinterpret_cast<char*>(naturalNeighborData->positions.pixels.data()), naturalNeighborData->positions.pixels.size() * sizeof(int));
    positionsFile.close();

    QFile indexMapFile(QString("nn.indexmap.%1.i32.bin").arg(BucketSize));
    indexMapFile.open(QIODevice::WriteOnly);
    indexMapFile.write(reinterpret_cast<char*>(naturalNeighborData->indexMap.pixels.data()), naturalNeighborData->indexMap.pixels.size() * sizeof(int));
    indexMapFile.close();

    QFile weightMapFile(QString("nn.weightmap.%1.f32.bin").arg(BucketSize));
    weightMapFile.open(QIODevice::WriteOnly);
    weightMapFile.write(reinterpret_cast<char*>(naturalNeighborData->weightMap.pixels.data()), naturalNeighborData->weightMap.pixels.size() * sizeof(float));
    weightMapFile.close();

    qInfo() << "Done";
}

void MainWindow::showEvent(QShowEvent* event) {
    m_imageViewer->fitInView(m_imageViewer->scene()->sceneRect(), Qt::KeepAspectRatio);
}

void MainWindow::updateImageViewer() {
    m_imageViewer->clear();

    if (!m_layers.isEmpty()) {
        const float voronoiScale = m_stippler.options().voronoiScale;
        QRect canvasArea = QTransform::fromScale(voronoiScale, voronoiScale).mapRect(m_layers[0].image.rect());
        m_imageViewer->setCanvasArea(canvasArea);

        if (m_toolbox->renderMode() == RenderMode::RasterStipples || m_toolbox->renderMode() == RenderMode::RasterBackground || m_toolbox->renderMode() == RenderMode::RasterStipplesWithBackground) {
            const float RenderScale = 1.0f;
            auto translateMode = [](RenderMode mode) {
                switch (mode) {
                case RenderMode::RasterStipples:
                    return RasterMode::Stipples;
                case RenderMode::RasterBackground:
                    return RasterMode::Background;
                case RenderMode::RasterStipplesWithBackground:
                    return RasterMode::StipplesWithBackground;
                default:
                    assert(false && "Unexpected render mode");
                    std::terminate();
                }
            };
            auto canvas = std::move(m_stippler.rasterize(RenderScale, translateMode(m_toolbox->renderMode())));
            QImage rasterizedImage(reinterpret_cast<const uchar*>(canvas.pixels.data()), canvas.width, canvas.height, QImage::Format_ARGB32);
            m_imageViewer->setBackground(rasterizedImage, RenderScale);
        } else if (m_toolbox->renderMode() == RenderMode::PainterStipples || m_toolbox->renderMode() == RenderMode::PainterHighlightedStipples) {
            QPoint stippleOffset(-m_stippler.options().borderWidth, -m_stippler.options().borderWidth);
            QSize stippleArea = canvasArea.size() + QSize(m_stippler.options().borderWidth * 2, m_stippler.options().borderWidth * 2);

            if (m_toolbox->renderMode() == RenderMode::PainterHighlightedStipples) {
                m_imageViewer->setHightlightSplits(true);
            } else {
                m_imageViewer->setHightlightSplits(false);
            }

            for (int i = 0; i < m_layers.size(); ++i) {
                if (m_layers[i].visible && m_layers[i].stipples.size() > 0) {
                    m_imageViewer->setStipples(i, m_layers[i].stipples, stippleArea, stippleOffset);
                }
            }
        } else if (m_toolbox->renderMode() == RenderMode::PainterDensity) {
            for (int i = 0; i < m_layers.size(); ++i) {
                if (m_layers[i].visible) {
                    QColor color(m_layers[i].stippler.lbg.color.r(), m_layers[i].stippler.lbg.color.g(), m_layers[i].stippler.lbg.color.b());
                    m_imageViewer->setDensity(i, m_layers[i].stippler.density, voronoiScale, color);
                }
            }
        }
    }
}

#if defined(QT_MULTIMEDIA_LIB)
FrameWorker::FrameWorker(bool& frameChanging, QObject* parent)
    : QObject(parent)
    , m_ready(true)
    , m_frameChanging(frameChanging) {
}

void FrameWorker::processVideoFrame(const QVideoFrame& frame) {
    if (!m_ready) {
        return;
    }
    m_ready = false;
    // qDebug() << "processVideoFrame";

    QImage image = frame.toImage().convertToFormat(QImage::Format_Grayscale8);
    image = image.scaled(image.size() / 2);

    std::vector<float> pixelsBlack;
    std::vector<float> pixelsWhite;
    pixelsBlack.reserve(image.width() * image.height());
    pixelsWhite.reserve(image.width() * image.height());
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            QRgb pixel = image.pixel(x, y);
            float density = (1.0f - qGray(pixel) / 255.0f) * qAlpha(pixel) / 255.0f;
            density = std::max(0.0f, std::min(1.0f, density));
            pixelsBlack.push_back(density);
            pixelsWhite.push_back(1.0f - density);
        }
    }

    std::vector<Map<float>> maps = {
        Map<float> {
            image.width(),
            image.height(),
            std::move(pixelsBlack) },
        Map<float> {
            image.width(),
            image.height(),
            std::move(pixelsWhite) }
    };

    emit mapsChanged(std::move(maps));

    // XXX: busy waiting... could be better.
    while (m_frameChanging) {
        QThread::msleep(1);
    }

    // Once finished wait a bit.
    // QThread::msleep(2000);

    m_ready = true;
}

#endif
