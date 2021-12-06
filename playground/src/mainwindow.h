#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "imageviewer.h"
#include "layers.h"
#include "stippler.h"
#include "toolboxwidget.h"
#include <QDir>
#include <QElapsedTimer>
#include <QJsonObject>
#include <QMainWindow>

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(Layers layers, QWidget* parent = nullptr);

    QJsonObject parametersToJson();
    void parametersFromJson(QJsonObject json, const QString* path = nullptr);

    void setCanvasColor(QColor color);

    void setIntermediateResultDisplay(int minIterationDuration);
    void setIterationCallback(Stippler::IterationCallback cb);

    void setOptions(StipplerOptions options);
    void setLayers(Layers layers);

public slots:
    void loadImages(const QStringList& paths);
    void loadImageAsDual(const QString& path);
    void loadLayers(Layers layers);

    void loadProject(const QString& path);
    void saveProject(const QString& path);
    void saveImage(const QString& path);

    qint64 stipple();

    void computeAndSaveNaturalNeighborData();

protected:
    void showEvent(QShowEvent* event) override;

private slots:
    void updateImageViewer();

private:
    Layers m_layers;
    Stippler m_stippler;

    QElapsedTimer m_stippleTimer;
    ImageViewer* m_imageViewer;
    ToolboxWidget* m_toolbox;
    QStatusBar* m_statusBar;
};

#endif
