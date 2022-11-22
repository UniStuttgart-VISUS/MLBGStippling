#ifndef TOOLBOXWIDGET_H
#define TOOLBOXWIDGET_H

#include "layers.h"
#include "metatypes.h"
#include "stippler.h"
#include <QScopedPointer>
#include <QWidget>
#if defined(QT_MULTIMEDIA_LIB)
#include <QCamera>
#endif

enum RenderMode {
    RasterStipplesWithBackground = 0,
    RasterStipples = 1,
    RasterBackground = 2,
    PainterStipples = 3,
    PainterHighlightedStipples = 4,
    PainterDensity = 5,
};

class ToolboxWidget : public QWidget {
    Q_OBJECT
public:
    explicit ToolboxWidget(QColor canvasColor, Layers& layers, Stippler& stippler, QWidget* parent = nullptr);

    void setEditable(bool editable);
    void setCanvasColor(const QColor color);
    RenderMode renderMode() const;
    void setRenderMode(RenderMode mode);
    void setMinIterationDuration(int minIterationDuration);

signals:
    void layerStyleChanged();
    void layerColorChanged();
    void canvasColorChanged(QColor canvasColor);
    void renderModeChanged(RenderMode mode);
    void minIterationDurationChanged(int durationMs);

    void start();
    void importAsDensity(const QStringList& paths);
    void importAsDual(const QString& path);
    void importLayers(Layers layers);
#if defined(QT_MULTIMEDIA_LIB)
    void importCamera(QCamera* camera);
#endif
    void exportImage(const QString& path);
    void exportNaturalNeighborData();
    void loadProject(const QString& path);
    void saveProject(const QString& path);

    // These are signals for internal implementation reasons, do not subscribe!
    void invalidateParameterWidgets();
    void invalidateLayerWidgets();

private:
    QWidget* createSettingsGroup();
    QWidget* createCommonSettingsGroup(QWidget* parent);
    QWidget* createLayerSettingsGroups(QWidget* parent);
    QWidget* createExpandableLayerWidget(QWidget* parent, size_t layerIndex);
    QWidget* createLayerSettingsGroup(QWidget* parent, size_t layerIndex);
    QWidget* createButtonGroup();

    QVector<QWidget*> m_editables;
    std::vector<std::unique_ptr<QWidget>> m_layerGroups;

    Layers& m_layers;
    Stippler& m_stippler;

    QColor m_canvasColor;
    RenderMode m_renderMode;
    int m_minIterationDuration;
};

#endif
