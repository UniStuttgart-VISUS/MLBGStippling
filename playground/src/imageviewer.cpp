#include "imageviewer.h"
#include <QOpenGLWidget>
#include <QPdfWriter>
#include <QSvgGenerator>
#include <QtWidgets>

uint qHash(QColor key) {
    return key.rgba();
}

uint qHash(std::tuple<QColor, float> key) {
    return std::get<0>(key).rgba() ^ qHash(std::get<1>(key));
}

class CircleStipplesItem : public QGraphicsItem {
public:
    CircleStipplesItem(const std::vector<Stipple>& stipples, QSizeF targetSize, QPointF offset, QGraphicsItem* parent = nullptr)
        : QGraphicsItem(parent)
        , m_boundingRect(QPointF(0, 0), targetSize) {
        for (const auto& stipple : stipples) {
            // Transform to view space.
            float size = stipple.size;
            auto x = stipple.center.x - size / 2.0f + offset.x();
            auto y = stipple.center.y - size / 2.0f + offset.y();
            QRectF rect(x, y, size, size);
            QColor color(stipple.color.r(), stipple.color.g(), stipple.color.b(), stipple.color.a());
            // Store it for rendering.
            if (!m_rects.contains(color)) {
                m_rects[color] = QVector<QRectF>();
            }
            m_rects[color].append(rect);
        }
    }

    QRectF boundingRect() const override { return m_boundingRect; }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override {
        QPen pen = painter->pen();
        QBrush brush = painter->brush();
        painter->setPen(Qt::NoPen);
        for (auto iter = m_rects.cbegin(); iter != m_rects.cend(); ++iter) {
            painter->setBrush(iter.key());
            for (const auto rect : iter.value()) {
                if (painter->paintEngine()->type() == QPaintEngine::OpenGL || painter->paintEngine()->type() == QPaintEngine::OpenGL2) {
                    // Drawing rounded rects is faster for these engines, but would result in
                    // paths for vector graphics backends.
                    qreal radius = rect.width() / 2.0f;
                    painter->drawRoundedRect(rect, radius, radius);
                } else {
                    painter->drawEllipse(rect);
                }
            }
        }
        painter->setPen(painter->pen());
        painter->setBrush(brush);
    }

protected:
    QRectF m_boundingRect;

    QHash<QColor, QVector<QRectF>> m_rects;
};

class LineStipplesItem : public QGraphicsItem {
public:
    LineStipplesItem(const std::vector<Stipple>& stipples, QSizeF targetSize, QPointF offset, QGraphicsItem* parent = nullptr)
        : QGraphicsItem(parent)
        , m_boundingRect(QPointF(0, 0), targetSize) {
        for (const auto& stipple : stipples) {
            // Transform to view space.
            float size = stipple.size;
            QPointF center = QPointF(stipple.center.x + offset.x(), stipple.center.y + offset.y());
            QPointF centerOffset = QPointF(cos(stipple.rotation), sin(stipple.rotation));
            centerOffset *= size / 2.0f;
            QColor color(stipple.color.r(), stipple.color.g(), stipple.color.b(), stipple.color.a());
            // Store it for rendering.
            BatchTuple tuple(color, stipple.shapeParameter); //XXX: this has the potential to generate alot of batches.
            if (!m_rects.contains(tuple)) {
                m_rects[tuple] = QVector<std::pair<QPointF, QPointF>>();
            }
            m_rects[tuple].append({ center - centerOffset, center + centerOffset });
        }
    }

    QRectF boundingRect() const override { return m_boundingRect; }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override {
        QPen pen = painter->pen();
        QBrush brush = painter->brush();
        for (auto iter = m_rects.cbegin(); iter != m_rects.cend(); ++iter) {
            auto [color, lineWidth] = iter.key();
            QPen tempPen = QPen(color, lineWidth, Qt::SolidLine);
            tempPen.setCapStyle(Qt::FlatCap);
            painter->setPen(tempPen);
            for (const auto line : iter.value()) {
                painter->drawLine(line.first, line.second);
            }
        }
        painter->setPen(pen);
        painter->setBrush(brush);
    }

protected:
    QRectF m_boundingRect;

    using BatchTuple = std::tuple<QColor, float>;

    QHash<BatchTuple, QVector<std::pair<QPointF, QPointF>>> m_rects;
};

class RectangleStipplesItem : public QGraphicsItem {
public:
    RectangleStipplesItem(const std::vector<Stipple>& stipples, QSizeF targetSize, QPointF offset, QGraphicsItem* parent = nullptr)
        : QGraphicsItem(parent)
        , m_boundingRect(QPointF(0, 0), targetSize) {
        for (const auto& stipple : stipples) {
            // Transform to view space.
            float size = stipple.size;
            QPointF center = QPointF(stipple.center.x, stipple.center.y) + offset;
            QPointF directionX = QPointF(cos(stipple.rotation), sin(stipple.rotation)) * size / 2.0f * stipple.shapeParameter;
            QPointF directionY = QPointF(-sin(stipple.rotation), cos(stipple.rotation)) * size / 2.0f;
            QColor color(stipple.color.r(), stipple.color.g(), stipple.color.b(), stipple.color.a());
            // Store it for rendering.
            if (!m_polys.contains(color)) {
                m_polys[color] = QVector<QVector<QPointF>>();
            }
            m_polys[color].append({ center + directionX + directionY, center - directionX + directionY, center - directionX - directionY, center + directionX - directionY });
        }
    }

    QRectF boundingRect() const override { return m_boundingRect; }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override {
        QPen pen = painter->pen();
        QBrush brush = painter->brush();
        painter->setPen(Qt::NoPen);
        for (auto iter = m_polys.cbegin(); iter != m_polys.cend(); ++iter) {
            painter->setBrush(iter.key());
            for (const auto poly : iter.value()) {
                painter->drawPolygon(&poly[0], 4);
            }
        }
        painter->setPen(painter->pen());
        painter->setBrush(brush);
    }

protected:
    QRectF m_boundingRect;

    QHash<QColor, QVector<QVector<QPointF>>> m_polys;
};

class RhombusStipplesItem : public QGraphicsItem {
public:
    RhombusStipplesItem(const std::vector<Stipple>& stipples, QSizeF targetSize, QPointF offset, QGraphicsItem* parent = nullptr)
        : QGraphicsItem(parent)
        , m_boundingRect(QPointF(0, 0), targetSize) {
        for (const auto& stipple : stipples) {
            // Transform to view space.
            float size = stipple.size;
            QPointF center = QPointF(stipple.center.x, stipple.center.y) + offset;
            QPointF directionX = QPointF(cos(stipple.rotation), sin(stipple.rotation)) * size / 2.0f * stipple.shapeParameter;
            QPointF directionY = QPointF(-sin(stipple.rotation), cos(stipple.rotation)) * size / 2.0f;
            QColor color(stipple.color.r(), stipple.color.g(), stipple.color.b(), stipple.color.a());
            // Store it for rendering.
            if (!m_polys.contains(color)) {
                m_polys[color] = QVector<QVector<QPointF>>();
            }
            m_polys[color].append({ center + directionX, center + directionY, center - directionX, center - directionY });
        }
    }

    QRectF boundingRect() const override { return m_boundingRect; }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override {
        QPen pen = painter->pen();
        QBrush brush = painter->brush();
        painter->setPen(Qt::NoPen);
        for (auto iter = m_polys.cbegin(); iter != m_polys.cend(); ++iter) {
            painter->setBrush(iter.key());
            for (const auto poly : iter.value()) {
                painter->drawPolygon(&poly[0], 4);
            }
        }
        painter->setPen(painter->pen());
        painter->setBrush(brush);
    }

protected:
    QRectF m_boundingRect;

    QHash<QColor, QVector<QVector<QPointF>>> m_polys;
};

class EllipseStipplesItem : public QGraphicsItem {
public:
    struct Ellipse {
        QPointF center;
        QPointF size;
        float angle;
    };

    EllipseStipplesItem(const std::vector<Stipple>& stipples, QSizeF targetSize, QPointF offset, QGraphicsItem* parent = nullptr)
        : QGraphicsItem(parent)
        , m_boundingRect(QPointF(0, 0), targetSize) {
        for (const auto& stipple : stipples) {
            // Transform to view space.
            QPointF center = QPointF(stipple.center.x, stipple.center.y) + offset;
            QPointF size = QPointF(stipple.size * stipple.shapeParameter, stipple.size);
            QColor color(stipple.color.r(), stipple.color.g(), stipple.color.b(), stipple.color.a());
            // Store it for rendering.
            if (!m_ellipses.contains(color)) {
                m_ellipses[color] = QVector<Ellipse>();
            }
            m_ellipses[color].append({ center, size, stipple.rotation });
        }
    }

    QRectF boundingRect() const override { return m_boundingRect; }

    void
    paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override {
        QPen pen = painter->pen();
        QBrush brush = painter->brush();
        painter->setPen(Qt::NoPen);
        for (auto iter = m_ellipses.cbegin(); iter != m_ellipses.cend(); ++iter) {
            painter->setBrush(iter.key());
            for (const auto Ellipse : iter.value()) {
                painter->save();
                painter->translate(Ellipse.center);
                painter->rotate(qRadiansToDegrees(Ellipse.angle));
                painter->drawEllipse(QRectF(-Ellipse.size / 2.0f, Ellipse.size / 2.0f));
                painter->restore();
            }
        }
        painter->setPen(painter->pen());
        painter->setBrush(brush);
    }

protected:
    QRectF m_boundingRect;

    QHash<QColor, QVector<Ellipse>> m_ellipses;
};

class TriangleStipplesItem : public QGraphicsItem {
public:
    TriangleStipplesItem(const std::vector<Stipple>& stipples, QSizeF targetSize, QPointF offset, QGraphicsItem* parent = nullptr)
        : QGraphicsItem(parent)
        , m_boundingRect(QPointF(0, 0), targetSize) {
        for (const auto& stipple : stipples) {
            // Transform to view space.
            float size = stipple.size;
            QPointF center = QPointF(stipple.center.x, stipple.center.y) + offset;
            float radius = size / std::sqrt(3);
            float angles[3] = { 0.0f, M_PI * 4.0f / 6.0f, M_PI * 8.0f / 6.0f };
            QVector<QPointF> cornerPoints = QVector<QPointF>();

            for (float angle : angles) {
                cornerPoints.append(center + QPointF(cos(stipple.rotation + angle), sin(stipple.rotation + angle)) * radius);
            }

            QColor color(stipple.color.r(), stipple.color.g(), stipple.color.b(), stipple.color.a());
            // Store it for rendering.
            if (!m_polys.contains(color)) {
                m_polys[color] = QVector<QVector<QPointF>>();
            }
            m_polys[color].append(cornerPoints);
        }
    }

    QRectF boundingRect() const override { return m_boundingRect; }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override {
        QPen pen = painter->pen();
        QBrush brush = painter->brush();
        painter->setPen(Qt::NoPen);
        for (auto iter = m_polys.cbegin(); iter != m_polys.cend(); ++iter) {
            painter->setBrush(iter.key());
            for (const auto poly : iter.value()) {
                painter->drawPolygon(&poly[0], 3);
            }
        }
        painter->setPen(painter->pen());
        painter->setBrush(brush);
    }

protected:
    QRectF m_boundingRect;

    QHash<QColor, QVector<QVector<QPointF>>> m_polys;
};

class RoundedLineStipplesItem : public QGraphicsItem {
public:
    RoundedLineStipplesItem(const std::vector<Stipple>& stipples, QSizeF targetSize, QPointF offset, QGraphicsItem* parent = nullptr)
        : QGraphicsItem(parent)
        , m_boundingRect(QPointF(0, 0), targetSize) {
        for (const auto& stipple : stipples) {
            // Transform to view space.
            float size = stipple.size;
            QPointF center = QPointF(stipple.center.x, stipple.center.y) + offset;
            QPointF directionX = QPointF(cos(stipple.rotation), sin(stipple.rotation)) * (size - stipple.shapeRadius * stipple.shapeParameter) / 2.0f;
            QPointF directionY = QPointF(-sin(stipple.rotation), cos(stipple.rotation)) * (1.0f - stipple.shapeRadius) * (stipple.shapeParameter / 2.0f);
            QVector<QPointF> cornerPoints = { center + directionX + directionY, center - directionX + directionY, center - directionX - directionY, center + directionX - directionY };
            QVector<QPointF> polyPoints = QVector<QPointF>();
            float segmentRadius = M_PI / ((pointsPerCorner - 1) * 2);
            for (int cornerIndex = 0; cornerIndex < 4; ++cornerIndex) {
                for (int segmentIndex = 0; segmentIndex < pointsPerCorner; ++segmentIndex) {
                    QPointF corner = cornerPoints[cornerIndex];
                    float angle = (cornerIndex * (pointsPerCorner - 1) + segmentIndex) * segmentRadius + stipple.rotation;
                    polyPoints.append(corner + QPointF(cos(angle), sin(angle)) * stipple.shapeRadius * stipple.shapeParameter / 2.0f);
                }
            }
            QColor color(stipple.color.r(), stipple.color.g(), stipple.color.b(), stipple.color.a());
            // Store it for rendering.
            if (!m_polys.contains(color)) {
                m_polys[color] = QVector<QVector<QPointF>>();
            }
            m_polys[color].append(polyPoints);
        }
    }

    QRectF boundingRect() const override { return m_boundingRect; }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override {
        QPen pen = painter->pen();
        QBrush brush = painter->brush();
        painter->setPen(Qt::NoPen);
        for (auto iter = m_polys.cbegin(); iter != m_polys.cend(); ++iter) {
            painter->setBrush(iter.key());
            for (const auto poly : iter.value()) {
                painter->drawPolygon(&poly[0], pointsPerCorner * 4);
            }
        }
        painter->setPen(painter->pen());
        painter->setBrush(brush);
    }

protected:
    QRectF m_boundingRect;
    int pointsPerCorner = 10;

    QHash<QColor, QVector<QVector<QPointF>>> m_polys;
};

class RoundedRectangleStipplesItem : public QGraphicsItem {
public:
    RoundedRectangleStipplesItem(const std::vector<Stipple>& stipples, QSizeF targetSize, QPointF offset, QGraphicsItem* parent = nullptr)
        : QGraphicsItem(parent)
        , m_boundingRect(QPointF(0, 0), targetSize) {
        for (const auto& stipple : stipples) {
            // Transform to view space.
            float size = stipple.size;
            QPointF center = QPointF(stipple.center.x, stipple.center.y) + offset;
            QPointF directionX = QPointF(cos(stipple.rotation), sin(stipple.rotation)) * (size / 2.0f) * (stipple.shapeParameter - stipple.shapeRadius);
            QPointF directionY = QPointF(-sin(stipple.rotation), cos(stipple.rotation)) * (size / 2.0f) * (1.0f - stipple.shapeRadius);
            QVector<QPointF> cornerPoints = { center + directionX + directionY, center - directionX + directionY, center - directionX - directionY, center + directionX - directionY };
            QVector<QPointF> polyPoints = QVector<QPointF>();
            float segmentRadius = M_PI / ((pointsPerCorner - 1) * 2);
            for (int cornerIndex = 0; cornerIndex < 4; ++cornerIndex) {
                for (int segmentIndex = 0; segmentIndex < pointsPerCorner; ++segmentIndex) {
                    QPointF corner = cornerPoints[cornerIndex];
                    float angle = (cornerIndex * (pointsPerCorner - 1) + segmentIndex) * segmentRadius + stipple.rotation;
                    polyPoints.append(corner + QPointF(cos(angle), sin(angle)) * stipple.shapeRadius * size / 2.0f);
                }
            }
            QColor color(stipple.color.r(), stipple.color.g(), stipple.color.b(), stipple.color.a());
            // Store it for rendering.
            if (!m_polys.contains(color)) {
                m_polys[color] = QVector<QVector<QPointF>>();
            }
            m_polys[color].append(polyPoints);
        }
    }

    QRectF boundingRect() const override { return m_boundingRect; }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override {
        QPen pen = painter->pen();
        QBrush brush = painter->brush();
        painter->setPen(Qt::NoPen);
        for (auto iter = m_polys.cbegin(); iter != m_polys.cend(); ++iter) {
            painter->setBrush(iter.key());
            for (const auto poly : iter.value()) {
                painter->drawPolygon(&poly[0], pointsPerCorner * 4);
            }
        }
        painter->setPen(painter->pen());
        painter->setBrush(brush);
    }

protected:
    QRectF m_boundingRect;
    int pointsPerCorner = 10;

    QHash<QColor, QVector<QVector<QPointF>>> m_polys;
};

class RoundedRhombusStipplesItem : public QGraphicsItem {
public:
    RoundedRhombusStipplesItem(const std::vector<Stipple>& stipples, QSizeF targetSize, QPointF offset, QGraphicsItem* parent = nullptr)
        : QGraphicsItem(parent)
        , m_boundingRect(QPointF(0, 0), targetSize) {
        for (const auto& stipple : stipples) {
            // Transform to view space.
            float size = (1.0f - stipple.shapeRadius) * stipple.size;
            QPointF center = QPointF(stipple.center.x, stipple.center.y) + offset;
            QPointF directionX = QPointF(cos(stipple.rotation), sin(stipple.rotation)) * size / 2.0f * stipple.shapeParameter;
            QPointF directionY = QPointF(-sin(stipple.rotation), cos(stipple.rotation)) * size / 2.0f;
            QVector<QPointF> cornerPoints = { center + directionX, center + directionY, center - directionX, center - directionY };
            QVector<QPointF> polyPoints = QVector<QPointF>();

            float angleOffset = atanf(stipple.shapeParameter);
            //inscribed circle radius = a*b/(sqrt(a*a+b*b)*2), where a = stipple.size and b = stipple.size * stipple.shapeParameter
            const float inscribed = stipple.shapeRadius * stipple.size * stipple.shapeParameter / (std::sqrt((1.0f + stipple.shapeParameter * stipple.shapeParameter)) * 2.0f);
            for (int cornerIndex = 0; cornerIndex < 4; ++cornerIndex) {
                float startAngle = cornerIndex * M_PI / 2.0f - angleOffset;
                float endAngle = cornerIndex * M_PI / 2.0f + angleOffset;
                float segmentRadius = (endAngle - startAngle) / (pointsPerCorner - 1);
                for (int segmentIndex = 0; segmentIndex < pointsPerCorner; ++segmentIndex) {
                    QPointF corner = cornerPoints[cornerIndex];
                    float angle = startAngle + segmentIndex * segmentRadius + stipple.rotation;
                    polyPoints.append(corner + QPointF(cos(angle), sin(angle)) * inscribed);
                }
                angleOffset = M_PI / 2.0f - angleOffset;
            }
            QColor color(stipple.color.r(), stipple.color.g(), stipple.color.b(), stipple.color.a());
            // Store it for rendering.
            if (!m_polys.contains(color)) {
                m_polys[color] = QVector<QVector<QPointF>>();
            }
            m_polys[color].append(polyPoints);
        }
    }

    QRectF boundingRect() const override { return m_boundingRect; }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override {
        QPen pen = painter->pen();
        QBrush brush = painter->brush();
        painter->setPen(Qt::NoPen);
        for (auto iter = m_polys.cbegin(); iter != m_polys.cend(); ++iter) {
            painter->setBrush(iter.key());
            for (const auto poly : iter.value()) {
                painter->drawPolygon(&poly[0], pointsPerCorner * 4);
            }
        }
        painter->setPen(painter->pen());
        painter->setBrush(brush);
    }

protected:
    QRectF m_boundingRect;
    int pointsPerCorner = 10;

    QHash<QColor, QVector<QVector<QPointF>>> m_polys;
};

class RoundedTriangleStipplesItem : public QGraphicsItem {
public:
    RoundedTriangleStipplesItem(const std::vector<Stipple>& stipples, QSizeF targetSize, QPointF offset, QGraphicsItem* parent = nullptr)
        : QGraphicsItem(parent)
        , m_boundingRect(QPointF(0, 0), targetSize) {
        for (const auto& stipple : stipples) {
            // Transform to view space.
            float triangle_side = (1.0f - stipple.shapeRadius) * stipple.size;
            QPointF center = QPointF(stipple.center.x, stipple.center.y) + offset;
            float radius = triangle_side / std::sqrt(3);
            float angles[3] = {
                stipple.rotation,
                (float)M_PI * 4.0f / 6.0f + stipple.rotation,
                (float)M_PI * 8.0f / 6.0f + stipple.rotation
            };
            QVector<QPointF> cornerPoints = QVector<QPointF>();

            for (float angle : angles) {
                cornerPoints.append(center + QPointF(cos(angle), sin(angle)) * radius);
            }

            QVector<QPointF> polyPoints = QVector<QPointF>();
            float segmentRadius = (2.0 * M_PI) / ((pointsPerCorner - 1) * 3);
            const float r_inscribed = stipple.shapeRadius * stipple.size * std::sqrt(3.0f) / 6.0f;
            for (int cornerIndex = 0; cornerIndex < 3; ++cornerIndex) {
                for (int segmentIndex = 0; segmentIndex < pointsPerCorner; ++segmentIndex) {
                    QPointF corner = cornerPoints[cornerIndex];
                    float angle = angles[(cornerIndex + 1) % 3] + M_PI + segmentIndex * segmentRadius;
                    polyPoints.append(corner + QPointF(cos(angle), sin(angle)) * r_inscribed);
                }
            }

            QColor color(stipple.color.r(), stipple.color.g(), stipple.color.b(), stipple.color.a());
            // Store it for rendering.
            if (!m_polys.contains(color)) {
                m_polys[color] = QVector<QVector<QPointF>>();
            }
            m_polys[color].append(polyPoints);
        }
    }

    QRectF boundingRect() const override { return m_boundingRect; }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override {
        QPen pen = painter->pen();
        QBrush brush = painter->brush();
        painter->setPen(Qt::NoPen);
        for (auto iter = m_polys.cbegin(); iter != m_polys.cend(); ++iter) {
            painter->setBrush(iter.key());
            for (const auto poly : iter.value()) {
                painter->drawPolygon(&poly[0], pointsPerCorner * 3);
            }
        }
        painter->setPen(painter->pen());
        painter->setBrush(brush);
    }

protected:
    QRectF m_boundingRect;
    int pointsPerCorner = 10;

    QHash<QColor, QVector<QVector<QPointF>>> m_polys;
};

ImageViewer::ImageViewer(QWidget* parent)
    : QGraphicsView(parent)
    , m_canvasColor(QColor(119, 119, 119)) {
    // OpenGL acceleration.
    QSurfaceFormat format = QSurfaceFormat::defaultFormat();
    format.setDepthBufferSize(0);
    format.setSamples(8);
    QOpenGLWidget* viewport = new QOpenGLWidget();
    viewport->setFormat(format);
    setViewport(viewport);

    // Rendering-related things.
    setRenderHint(QPainter::Antialiasing, true);
    setRenderHint(QPainter::SmoothPixmapTransform, true);
    setOptimizationFlag(QGraphicsView::DontSavePainterState);
    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    setCacheMode(QGraphicsView::CacheNone);

    // Look and feel.
    setInteractive(false);
    setDragMode(QGraphicsView::ScrollHandDrag);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    setBackgroundBrush(palette().brush(QPalette::Dark));
    setAlignment(Qt::AlignCenter);
    setMinimumSize(QSize(480, 640));
    setFrameStyle(0);

    setScene(new QGraphicsScene(this));
}

void ImageViewer::clear() {
    this->scene()->clear();
    m_stipplesSplit = 0;
    m_stipplesTotal = 0;
}

void ImageViewer::setCanvasArea(const QRectF& rect) {
    this->scene()->setSceneRect(rect);
}

QColor ImageViewer::canvasColor() const {
    return m_canvasColor;
}

void ImageViewer::setCanvasColor(QColor color) {
    m_canvasColor = color;
    this->update();
}

void ImageViewer::setHightlightSplits(bool enabled) {
    m_highlightSplits = enabled;
}

void ImageViewer::setBackground(const QImage& image, float scale) {
    QGraphicsPixmapItem* item = this->scene()->addPixmap(QPixmap::fromImage(image));
    item->setTransformationMode(Qt::SmoothTransformation);
    item->setScale(1.0f / scale);
    item->setZValue(-2.0f);
}

void ImageViewer::setDensity(int layerIndex, const Map<float>& density, float scale, const QColor& color) {
    QImage image(density.width, density.height, QImage::Format_ARGB32);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            int alpha = std::min(255.0f, std::max(0.0f, density.pixels[y * density.width + x] * 255.0f));
            QRgb newPixel = qRgba(color.red(), color.green(), color.blue(), alpha);
            image.setPixel(x, y, newPixel);
        }
    }

    QGraphicsPixmapItem* item = this->scene()->addPixmap(QPixmap::fromImage(image));
    item->setTransformationMode(Qt::SmoothTransformation);
    item->setScale(scale);
    item->setZValue(-1.0f);
}

void ImageViewer::setStipples(int layerIndex,
    const std::vector<Stipple>& stipples,
    QSize imageSize, QPointF offset) {

    m_stipplesSplit += std::count_if(stipples.begin(), stipples.end(),
        [](const auto& stipple) { return stipple.state == StippleState::New; });
    m_stipplesTotal += stipples.size();

    std::vector<Stipple> stipplesCopy = stipples;
    if (m_highlightSplits) {
        int h, s, v;
        for (auto& stipple : stipplesCopy) {
            if (stipple.state == StippleState::New) {
                QColor(stipple.color.argb).getHsv(&h, &s, &v);
                s = (s + 80 * (s < 128 ? 1 : -1));
                v = (v + 80 * (v < 128 ? 1 : -1));
                QColor highlightColor = QColor::fromHsv(h, s, v);
                stipple.color = Color(highlightColor.red(), highlightColor.green(), highlightColor.blue());
            } else if (stipple.state == StippleState::Merged) {
                stipple.color = Color(255, 0, 255);
            }
        }
    }

    const auto newStippleItem = [this, imageSize, offset](const std::vector<Stipple>& stipples) -> QGraphicsItem* {
        switch (stipples[0].shape) {
        case StippleShape::Circle:
            return new CircleStipplesItem(stipples, imageSize, offset);
        case StippleShape::Line:
            return new LineStipplesItem(stipples, imageSize, offset);
        case StippleShape::Rectangle:
            return new RectangleStipplesItem(stipples, imageSize, offset);
        case StippleShape::Rhombus:
            return new RhombusStipplesItem(stipples, imageSize, offset);
        case StippleShape::Ellipse:
            return new EllipseStipplesItem(stipples, imageSize, offset);
        case StippleShape::Triangle:
            return new TriangleStipplesItem(stipples, imageSize, offset);
        case StippleShape::RoundedLine:
            return new RoundedLineStipplesItem(stipples, imageSize, offset);
        case StippleShape::RoundedRectangle:
            return new RoundedRectangleStipplesItem(stipples, imageSize, offset);
        case StippleShape::RoundedRhombus:
            return new RoundedRhombusStipplesItem(stipples, imageSize, offset);
        case StippleShape::RoundedTriangle:
            return new RoundedTriangleStipplesItem(stipples, imageSize, offset);
        default:
            Q_UNREACHABLE();
        }
    };

    if (!stipplesCopy.empty()) {
        auto item = newStippleItem(stipplesCopy);
        this->scene()->addItem(item);
    }
}

void ImageViewer::save(const QString& path) {
    if (path.endsWith("svg", Qt::CaseInsensitive)) {
        this->saveSVG(path);
    } else if (path.endsWith("pdf", Qt::CaseInsensitive)) {
        this->savePDF(path);
    } else {
        this->savePixmap(path);
    }
}

void ImageViewer::drawBackground(QPainter* painter, const QRectF& rect) {
    QGraphicsView::drawBackground(painter, rect);
    painter->fillRect(this->sceneRect(), m_canvasColor);
}

void ImageViewer::wheelEvent(QWheelEvent* event) {
    QGraphicsView::wheelEvent(event);
    if (event->angleDelta().y() > 0) {
        this->scale(1.25, 1.25);
    } else {
        this->scale(0.8, 0.8);
    }
}

void ImageViewer::savePDF(const QString& path) {
    QPdfWriter writer(path);
    writer.setCreator(QApplication::applicationName());
    writer.setPageSize(QPageSize(this->scene()->sceneRect().size(), QPageSize::Point));
    writer.setPageMargins(QMarginsF());

    QPainter painter(&writer);
    this->scene()->render(&painter);
    painter.end();
}

void ImageViewer::saveSVG(const QString& path) {
    QSvgGenerator generator;
    generator.setFileName(path);
    generator.setSize(QSize(this->scene()->width(), this->scene()->height()));
    generator.setViewBox(this->scene()->sceneRect());
    generator.setDescription(QString("Created by %1").arg(QApplication::applicationName()));

    QPainter painter(&generator);
    this->scene()->render(&painter);
    painter.end();
}

void ImageViewer::savePixmap(const QString& path) {
    QPixmap pixmap = this->toPixmap(1.0f);
    if (path.endsWith(".jpg")) { //XXX: do this properly
        QPixmap otherPixmap(pixmap.size());
        otherPixmap.fill(m_canvasColor);
        QPainter painter(&otherPixmap);
        painter.drawPixmap(0, 0, pixmap);
        painter.end();
        pixmap = otherPixmap;
    }
    pixmap.save(path);
}

QPixmap ImageViewer::toPixmap(float superSamplingFactor) {
    QRectF sceneRect = QTransform::fromScale(superSamplingFactor, superSamplingFactor)
                           .mapRect(this->scene()->sceneRect());

    QPixmap pixmap(sceneRect.size().toSize());
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
    painter.setClipRect(sceneRect);
    this->scene()->render(&painter);
    return std::move(pixmap);
}

void ImageViewer::contextMenuEvent(QContextMenuEvent* event) {
    const auto copyPixmap = [this](float superSamplingFactor, bool useAlpha) {
        return [this, superSamplingFactor, useAlpha](bool checked) {
            QPixmap pixmap = this->toPixmap(superSamplingFactor);
            QImage::Format format = useAlpha ? QImage::Format_ARGB32 : QImage::Format_RGB32;
            QImage image(pixmap.size(), format);
            if (useAlpha) {
                image.fill(Qt::transparent);
            } else {
                image.fill(m_canvasColor);
            }
            QPainter painter(&image);
            painter.drawPixmap(0, 0, pixmap);
            // Workaround since clipboard()->setImage() does not work with GIMP, Word, etc.
            QByteArray byteArray;
            QBuffer buffer(&byteArray);
            buffer.open(QIODevice::WriteOnly);
            image.save(&buffer, "PNG");
            buffer.close();
            QMimeData* mime = new QMimeData();
            mime->setImageData(image);
            mime->setData("PNG", byteArray);
            mime->setData("image/png", byteArray);
            QApplication::clipboard()->setMimeData(mime);
        };
    };
    QMenu menu(this);
    QAction* copyAction100 = menu.addAction("Copy at 100%");
    QAction* copyAction200 = menu.addAction("Copy at 200%");
    QAction* copyAction400 = menu.addAction("Copy at 400%");
    QAction* copyAction100Alpha = menu.addAction("Copy at 100% (transparent)");
    QAction* copyAction200Alpha = menu.addAction("Copy at 200% (transparent)");
    QAction* copyAction400Alpha = menu.addAction("Copy at 400% (transparent)");
    connect(copyAction100, QOverload<bool>::of(&QAction::triggered), copyPixmap(1.0f, false));
    connect(copyAction200, QOverload<bool>::of(&QAction::triggered), copyPixmap(2.0f, false));
    connect(copyAction400, QOverload<bool>::of(&QAction::triggered), copyPixmap(4.0f, false));
    connect(copyAction100Alpha, QOverload<bool>::of(&QAction::triggered), copyPixmap(1.0f, true));
    connect(copyAction200Alpha, QOverload<bool>::of(&QAction::triggered), copyPixmap(2.0f, true));
    connect(copyAction400Alpha, QOverload<bool>::of(&QAction::triggered), copyPixmap(4.0f, true));
    if (m_stipplesTotal > 0) {
        if (m_stipplesSplit > 1) {
            menu.addAction(QString("Stipples %L1 (%2 splits)").arg(m_stipplesTotal).arg(m_stipplesSplit))->setEnabled(false);
        } else {
            menu.addAction(QString("Stipples %L1").arg(m_stipplesTotal))->setEnabled(false);
        }
    }
    menu.addAction(QString("Position %1 , %2").arg(event->pos().x()).arg(event->pos().y()))->setEnabled(false);
    menu.exec(event->globalPos());
}
