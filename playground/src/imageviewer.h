#ifndef STIPPLEVIEWER_H
#define STIPPLEVIEWER_H

#include "stippler.h"
#include <QGraphicsView>
#include <QHash>
#include <QWheelEvent>

class ImageViewer : public QGraphicsView {
    Q_OBJECT
public:
    explicit ImageViewer(QWidget* parent = nullptr);

    void clear();

    void setCanvasArea(const QRectF& rect);

    QColor canvasColor() const;
    void setCanvasColor(QColor color);

    void setHightlightSplits(bool enabled);

    void setBackground(const QImage& image, float scale);

    void setDensity(int layerIndex, const Map<float>& map, float scale, const QColor& color);

    void setStipples(int layerIndex, const std::vector<Stipple>& stipples, QSize imageSize, QPointF offset);

    void save(const QString& path);

protected:
    void drawBackground(QPainter* painter, const QRectF& rect) override;

    void wheelEvent(QWheelEvent* event) override;

    void contextMenuEvent(QContextMenuEvent* event) override;

private:
    void savePDF(const QString& path);
    void saveSVG(const QString& path);
    void savePixmap(const QString& path);
    QPixmap toPixmap(float superSamplingFactor);

    QColor m_canvasColor;
    bool m_highlightSplits = false;
    int m_stipplesSplit = 0;
    int m_stipplesTotal = 0;
};

#endif
