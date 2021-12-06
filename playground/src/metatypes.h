#ifndef METATYPES_H
#define METATYPES_H

#include "layers.h"
#include "stippler.h"
#include <QMetaType>

Q_DECLARE_METATYPE(StippleShape)
Q_DECLARE_METATYPE(StippleAlgorithm)
Q_DECLARE_METATYPE(VoronoiAlgorithm)
Q_DECLARE_METATYPE(SizeFunction)
Q_DECLARE_METATYPE(StippleModel)
Q_DECLARE_METATYPE(SizeModel)
Q_DECLARE_METATYPE(HysteresisFunction)

inline QString stippleShapeToString(StippleShape stippleShape) {
    switch (stippleShape) {
    case StippleShape::Circle:
        return QLatin1String("Circle");
    case StippleShape::Line:
        return QLatin1String("Line");
    case StippleShape::Rectangle:
        return QLatin1String("Rectangle");
    case StippleShape::Rhombus:
        return QLatin1String("Rhombus");
    case StippleShape::Ellipse:
        return QLatin1String("Ellipse");
    case StippleShape::Triangle:
        return QLatin1String("Triangle");
    case StippleShape::RoundedLine:
        return QLatin1String("RoundedLine");
    case StippleShape::RoundedRectangle:
        return QLatin1String("RoundedRectangle");
    case StippleShape::RoundedRhombus:
        return QLatin1String("RoundedRhombus");
    case StippleShape::RoundedTriangle:
        return QLatin1String("RoundedTriangle");
    };
    return QString();
}

inline QString stippleAlgorithmToString(StippleAlgorithm stippleAlgorithm) {
    switch (stippleAlgorithm) {
    case StippleAlgorithm::LBG:
        return QLatin1String("LBG");
    case StippleAlgorithm::CoupledLBG:
        return QLatin1String("CoupledLBG");
    };
    return QString();
}

inline QString voronoiAlgorithmToString(VoronoiAlgorithm voronoiAlgorithm) {
    switch (voronoiAlgorithm) {
    case VoronoiAlgorithm::Search:
        return QLatin1String("Search");
    case VoronoiAlgorithm::JumpFlooding:
        return QLatin1String("JumpFlooding");
    };
    return QString();
}

inline QString sizeFunctionToString(SizeFunction sizeFunction) {
    switch (sizeFunction) {
    case SizeFunction::Linear:
        return QLatin1String("Linear");
    case SizeFunction::QuadraticIn:
        return QLatin1String("QuadraticIn");
    case SizeFunction::QuadraticOut:
        return QLatin1String("QuadraticOut");
    case SizeFunction::QuadraticInOut:
        return QLatin1String("QuadraticInOut");
    case SizeFunction::ExponentialIn:
        return QLatin1String("ExponentialIn");
    case SizeFunction::ExponentialOut:
        return QLatin1String("ExponentialOut");
    case SizeFunction::ExponentialInOut:
        return QLatin1String("ExponentialInOut");
    };
    return QString();
}

inline QString stippleModelToString(StippleModel stippleModel) {
    switch (stippleModel) {
    case StippleModel::Difference:
        return QLatin1String("Difference");
    case StippleModel::Convolution:
        return QLatin1String("Convolution");
    case StippleModel::ConvolutionFilling:
        return QLatin1String("ConvolutionFilling");
    case StippleModel::ConvolutionPacking:
        return QLatin1String("ConvolutionPacking");
    };
    return QString();
}

inline QString sizeModelToString(SizeModel sizeModel) {
    switch (sizeModel) {
    case SizeModel::Average:
        return QLatin1String("Average");
    case SizeModel::AdjustedAverage:
        return QLatin1String("AdjustedAverage");
    };
    return QString();
}

inline QString hysteresisFunctionToString(HysteresisFunction hysteresisFunction) {
    switch (hysteresisFunction) {
    case HysteresisFunction::Constant:
        return QLatin1String("Constant");
    case HysteresisFunction::Linear:
        return QLatin1String("Linear");
    case HysteresisFunction::LinearNoMSE:
        return QLatin1String("LinearNoMSE");
    case HysteresisFunction::ExponentialNoMSE:
        return QLatin1String("ExponentialNoMSE");
    };
    return QString();
}

inline StippleShape stringToStippleShape(QString stippleShape) {
    if (stippleShape.compare(QLatin1String("Circle")) == 0) {
        return StippleShape::Circle;
    } else if (stippleShape.compare(QLatin1String("Line")) == 0) {
        return StippleShape::Line;
    } else if (stippleShape.compare(QLatin1String("Rectangle")) == 0) {
        return StippleShape::Rectangle;
    } else if (stippleShape.compare(QLatin1String("Rhombus")) == 0) {
        return StippleShape::Rhombus;
    } else if (stippleShape.compare(QLatin1String("Ellipse")) == 0) {
        return StippleShape::Ellipse;
    } else if (stippleShape.compare(QLatin1String("Triangle")) == 0) {
        return StippleShape::Triangle;
    } else if (stippleShape.compare(QLatin1String("RoundedLine")) == 0) {
        return StippleShape::RoundedLine;
    } else if (stippleShape.compare(QLatin1String("RoundedRectangle")) == 0) {
        return StippleShape::RoundedRectangle;
    } else if (stippleShape.compare(QLatin1String("RoundedRhombus")) == 0) {
        return StippleShape::RoundedRhombus;
    } else if (stippleShape.compare(QLatin1String("RoundedTriangle")) == 0) {
        return StippleShape::RoundedTriangle;
    }
    return StippleShape::Circle;
}

inline StippleAlgorithm stringToStippleAlgorithm(QString stippleAlgorithm) {
    if (stippleAlgorithm.compare(QLatin1String("LBG")) == 0) {
        return StippleAlgorithm::LBG;
    } else if (stippleAlgorithm.compare(QLatin1String("CoupledLBG")) == 0) {
        return StippleAlgorithm::CoupledLBG;
    }
    return StippleAlgorithm::LBG;
}

inline VoronoiAlgorithm stringToVoronoiAlgorithm(QString voronoiAlgorithm) {
    if (voronoiAlgorithm.compare(QLatin1String("Search")) == 0) {
        return VoronoiAlgorithm::Search;
    } else if (voronoiAlgorithm.compare(QLatin1String("JumpFlooding")) == 0) {
        return VoronoiAlgorithm::JumpFlooding;
    }
    return VoronoiAlgorithm::Search;
}

inline SizeFunction stringToSizeFunction(QString sizeFunction) {
    if (sizeFunction.compare(QLatin1String("Linear")) == 0) {
        return SizeFunction::Linear;
    } else if (sizeFunction.compare(QLatin1String("QuadraticIn")) == 0) {
        return SizeFunction::QuadraticIn;
    } else if (sizeFunction.compare(QLatin1String("QuadraticOut")) == 0) {
        return SizeFunction::QuadraticOut;
    } else if (sizeFunction.compare(QLatin1String("QuadraticInOut")) == 0) {
        return SizeFunction::QuadraticInOut;
    } else if (sizeFunction.compare(QLatin1String("ExponentialIn")) == 0) {
        return SizeFunction::ExponentialIn;
    } else if (sizeFunction.compare(QLatin1String("ExponentialOut")) == 0) {
        return SizeFunction::ExponentialOut;
    } else if (sizeFunction.compare(QLatin1String("ExponentialInOut")) == 0) {
        return SizeFunction::ExponentialInOut;
    }
    return SizeFunction::Linear;
}

inline StippleModel stringToStippleModel(QString stippleModel) {
    if (stippleModel.compare(QLatin1String("Difference")) == 0) {
        return StippleModel::Difference;
    } else if (stippleModel.compare(QLatin1String("Convolution")) == 0) {
        return StippleModel::Convolution;
    } else if (stippleModel.compare(QLatin1String("ConvolutionPacking")) == 0
        || stippleModel.compare(QLatin1String("ConvolutionCapacityConstraining")) == 0) {
        return StippleModel::ConvolutionPacking;
    } else if (stippleModel.compare(QLatin1String("ConvolutionFilling")) == 0
        || stippleModel.compare(QLatin1String("ConvolutionDensityReinforcing")) == 0) {
        return StippleModel::ConvolutionFilling;
    }
    return StippleModel::Difference;
}

inline SizeModel stringToSizeModel(QString sizeModel) {
    if (sizeModel.compare(QLatin1String("Average")) == 0) {
        return SizeModel::Average;
    } else if (sizeModel.compare(QLatin1String("AdjustedAverage")) == 0) {
        return SizeModel::AdjustedAverage;
    }
    return SizeModel::Average;
}

inline HysteresisFunction stringToHysteresisFunction(QString hysteresisFunction) {
    if (hysteresisFunction.compare(QLatin1String("Constant")) == 0) {
        return HysteresisFunction::Constant;
    } else if (hysteresisFunction.compare(QLatin1String("Linear")) == 0) {
        return HysteresisFunction::Linear;
    } else if (hysteresisFunction.compare(QLatin1String("LinearNoMSE")) == 0) {
        return HysteresisFunction::LinearNoMSE;
    } else if (hysteresisFunction.compare(QLatin1String("Exponential")) == 0
        || hysteresisFunction.compare(QLatin1String("ExponentialNoMSE")) == 0) {
        return HysteresisFunction::ExponentialNoMSE;
    }
    return HysteresisFunction::Constant;
}

inline void registerConverters() {
    QMetaType::registerConverter<StippleShape, QString>(stippleShapeToString);
    QMetaType::registerConverter<StippleAlgorithm, QString>(stippleAlgorithmToString);
    QMetaType::registerConverter<VoronoiAlgorithm, QString>(voronoiAlgorithmToString);
    QMetaType::registerConverter<SizeFunction, QString>(sizeFunctionToString);
    QMetaType::registerConverter<StippleModel, QString>(stippleModelToString);
    QMetaType::registerConverter<SizeModel, QString>(sizeModelToString);
    QMetaType::registerConverter<HysteresisFunction, QString>(hysteresisFunctionToString);
    QMetaType::registerConverter<QString, StippleShape>(stringToStippleShape);
    QMetaType::registerConverter<QString, StippleAlgorithm>(stringToStippleAlgorithm);
    QMetaType::registerConverter<QString, VoronoiAlgorithm>(stringToVoronoiAlgorithm);
    QMetaType::registerConverter<QString, SizeFunction>(stringToSizeFunction);
    QMetaType::registerConverter<QString, StippleModel>(stringToStippleModel);
    QMetaType::registerConverter<QString, SizeModel>(stringToSizeModel);
    QMetaType::registerConverter<QString, HysteresisFunction>(stringToHysteresisFunction);
}

#endif
