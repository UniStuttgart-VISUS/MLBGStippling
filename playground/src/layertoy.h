#ifndef LAYERTOY_H
#define LAYERTOY_H

#include "layers.h"

inline QImage rasterizeImage(int width, int height, std::function<float(float, float)> fragmentCb) {
    QImage image(width, height, QImage::Format_Grayscale16);
    const float xStep = 1.0f / image.width();
    const float yStep = 1.0f / image.height();
    for (int y = 0; y < image.height(); ++y) {
        quint16* scanline = reinterpret_cast<quint16*>(image.bits() + y * image.bytesPerLine());
        for (int x = 0; x < image.width(); ++x) {
            scanline[x] = std::clamp(fragmentCb(x * xStep, y * yStep), 0.0f, 1.0f) * std::numeric_limits<quint16>::max();
        }
    }
    return image;
}

inline Layers grayscaleTestLayers() {
    auto stepIntensity = [](float u, float v) -> float {
        auto band = [](float u, float v) -> float {
            const int p[] = { 4, 8, 16, 32, 256, 256, 32, 16, 8, 4 };
            return static_cast<float>(p[static_cast<int>(v * 10.0f)]);
        };
        if (band(u, v) == 256.0f) {
            return 1.0f - u;
        } else {
            const float p = band(u, v);
            return 1.0f - std::floor(u * p) / (p - 1.0f);
        }
    };

    QImage blackImage = rasterizeImage(512, 32 * 10, [&](float u, float v) -> float {
        return stepIntensity(u, v);
    });
    QImage whiteImage = rasterizeImage(512, 32 * 10, [&](float u, float v) -> float {
        if (v >= 0.5f) {
            return 1.0f;
        } else {
            return 1.0f - stepIntensity(u, v);
        }
    });

    return { Layer(blackImage, "Black", Qt::black),
        Layer(whiteImage, "White", Qt::white) };
}

inline Layers uniformLayers(int count, float value) {
    Layers layers;
    for (int i = 0; i < count; ++i) {
        QImage image = rasterizeImage(
            256, 256, [=](float u, float _) -> float {
                return value;
            });
        layers.append(Layer(image, QString("Layer %1").arg(i), Qt::black));
    }
    return layers;
}

inline Layers invertedGradientLayers(bool allTransitions) {
    auto intensity = [](float u, float v) -> float {
        float x;
        if (u < 0.1f) {
            x = 1.0f;
        } else if (u < 0.9f) {
            const float uu = (u - 0.1f) / 0.8f;
            x = 0.5f - uu / 2.0f;
        } else {
            x = 0.0f;
        }

        if (v >= 0.5f) {
            return 0.5f - x;
        } else {
            return 0.5f + x;
        }
    };
    QImage blackImage = rasterizeImage(256, 64, [&](float u, float v) -> float {
        if (allTransitions) {
            return intensity(u, v);
        } else {
            return 1.0f - u;
        }
    });
    QImage whiteImage = rasterizeImage(256, 64, [&](float u, float v) -> float {
        if (allTransitions) {
            return 1.0f - intensity(u, v);
        } else {
            return u;
        }
    });

    return { Layer(blackImage, "Black", Qt::black),
        Layer(whiteImage, "White", Qt::white) };
}

inline Layers linearGradientLayer() {
    QImage blackImage = rasterizeImage(256, 64, [](float u, float v) -> float {
        (void)v;
        return 1.0f - u;
    });
    return { Layer(blackImage, "Black", Qt::black) };
}

inline Layers gaussianFoveaSamplingLayer() {
    const QSize screenSizePx(1920, 1080);
    const QSizeF screenSizeCm(53.35, 30.1);
    const float viewDistanceCm = 60;
    const float foveaAlpha = 5.0 / 180.0 * M_PI;
    const float gaussFactor = 0.7;
    const float foveaCm = viewDistanceCm * std::sin(foveaAlpha);
    const QSizeF foveaPx(screenSizePx.width() / screenSizeCm.width() * foveaCm,
        screenSizePx.height() / screenSizeCm.height() * foveaCm);

    auto ellipticalGauss2DAppox = [](float x, float y,
                                      float sigmaX, float sigmaY) -> float {
        return std::expf(-((x * x) / (2.0 * sigmaX * sigmaX) + (y * y) / (2.0 * sigmaY * sigmaY)));
    };

    QImage gaussianImage(screenSizePx.width() * 2, screenSizePx.height() * 2, QImage::Format_Grayscale8);
    for (int y = 0; y < gaussianImage.height(); ++y) {
        uchar* line = gaussianImage.scanLine(y);
        for (int x = 0; x < gaussianImage.width(); ++x) {
            float g = ellipticalGauss2DAppox(x - gaussianImage.width() / 2, y - gaussianImage.height() / 2, //
                foveaPx.width() * gaussFactor, foveaPx.height() * gaussFactor);
            line[x] = std::min(static_cast<int>((1.0f - g) * 255.0f), 254);
        }
    }

    Layer layer(gaussianImage, "Gaussian Fovea", Qt::black);
    layer.stippler.lbg.sizeMin = 1;
    layer.stippler.lbg.sizeMax = 1;

    return { layer };
}

#endif
