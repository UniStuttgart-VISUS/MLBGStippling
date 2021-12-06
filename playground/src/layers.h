#ifndef LAYERS_H
#define LAYERS_H

#include "stippler.h"
#include <QImage>
#include <QString>
#include <QVector>

struct Layer {
    Layer() = default;

    explicit Layer(QImage image, QString imagePath, QColor color)
        : image(image)
        , imagePath(imagePath) {
        setColor(color);
    }

    void setColor(QColor color) {
        stippler.lbg.color = Color(color.red(), color.green(), color.blue(), color.alpha());
    }

    bool visible = true;

    QImage image;
    QString imagePath;

    std::vector<Stipple> stipples;

    StipplerLayer stippler;
};

using Layers = QVector<Layer>;

#endif
