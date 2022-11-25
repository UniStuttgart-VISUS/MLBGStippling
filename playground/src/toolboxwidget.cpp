#include "toolboxwidget.h"
#include "clickablelabel.h"
#include "layertoy.h"
#include <QtWidgets>

#if defined(QT_MULTIMEDIA_LIB)
#include <QtMultimedia>
#endif

class ExpandableWidgetHeader : public QFrame {
public:
    const QSize iconSize = { 14, 14 };
    const QSize iconSizeWidget = { 16, 16 };
    explicit ExpandableWidgetHeader(QWidget* bodyWidget, const QString& text, const QIcon& icon, QWidget* parent = nullptr)
        : QFrame(parent)
        , m_bodyWidget(bodyWidget) {
        m_iconLabel = new QLabel(this);
        m_iconLabel->setPixmap(icon.pixmap(iconSize));
        m_iconLabel->setAlignment(Qt::AlignCenter);
        m_iconLabel->setFixedSize(iconSizeWidget);
        m_iconLabel->setMargin(1);

        m_textLabel = new ClickableLabel(text, this);
        connect(m_textLabel, &ClickableLabel::clicked, [this]() {
            m_bodyWidget->setHidden(!m_bodyWidget->isHidden());
            updateStateIcon();
        });
        m_textLabel->setMargin(5);

        m_stateLabel = new ClickableLabel("", this);
        m_stateLabel->setAlignment(Qt::AlignCenter);
        m_stateLabel->setFixedSize(iconSizeWidget);
        m_stateLabel->setMargin(1);
        connect(m_stateLabel, &ClickableLabel::clicked, [this]() {
            m_bodyWidget->setHidden(!m_bodyWidget->isHidden());
            updateStateIcon();
        });
        updateStateIcon();

        QHBoxLayout* layout = new QHBoxLayout(this);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->addWidget(m_stateLabel);
        layout->addWidget(m_iconLabel);
        layout->addWidget(m_textLabel, 1);

        setFrameStyle(QFrame::Panel | QFrame::Raised);
    };
    virtual ~ExpandableWidgetHeader() = default;

    void updateIconLabel(const QIcon& icon) {
        m_iconLabel->setPixmap(icon.pixmap(iconSize));
    }

    void updateTextLabel(const QString& text) {
        m_textLabel->setText(text);
    }

    QWidget* m_bodyWidget;

private:
    void updateStateIcon() {
        QIcon icon;
        if (m_bodyWidget->isHidden()) {
            icon = style()->standardIcon(QStyle::SP_ToolBarHorizontalExtensionButton);
        } else {
            icon = style()->standardIcon(QStyle::SP_ToolBarVerticalExtensionButton);
        }
        m_stateLabel->setPixmap(icon.pixmap(iconSize));
    }

    QLabel* m_iconLabel;
    ClickableLabel* m_textLabel;
    ClickableLabel* m_stateLabel;
};

QWidget* createExpandableWidget(QWidget* parent, QWidget* bodyWidget, const QString& text, const QIcon& icon, ExpandableWidgetHeader** headerPointer = nullptr) {
    QWidget* container = new QWidget(parent);
    QVBoxLayout* containerLayout = new QVBoxLayout(container);
    container->setLayout(containerLayout);
    containerLayout->setContentsMargins(0, 0, 0, 0);

    bodyWidget->setParent(container);
    ExpandableWidgetHeader* header = new ExpandableWidgetHeader(bodyWidget, text, icon, container);

    containerLayout->addWidget(header);
    containerLayout->addWidget(bodyWidget);

    return container;
}

QPixmap colorPixmap(const Color& color, QSize size) {
    QPixmap colorPixmap(size);
    colorPixmap.fill(QColor(color.r(), color.g(), color.b()));
    return std::move(colorPixmap);
}

QPixmap colorPixmap(const Color& color, const QPushButton* button) {
    Q_CHECK_PTR(button);
    return std::move(colorPixmap(color, QSize(180, 15)));
}

Color colorDialog(const Color& color) {
    QColor cIn(color.r(), color.g(), color.b());
    QColor cOut = QColorDialog::getColor(cIn);
    if (!cOut.isValid()) {
        cOut = cIn;
    }
    return Color(cOut.red(), cOut.green(), cOut.blue(), cOut.alpha());
}

ToolboxWidget::ToolboxWidget(QColor canvasColor, Layers& layers, Stippler& stippler, QWidget* parent)
    : QWidget(parent)
    , m_layers(layers)
    , m_stippler(stippler)
    , m_canvasColor(canvasColor)
    , m_renderMode(RenderMode::PainterDensity)
    , m_minIterationDuration(0) {
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    setLayout(layout);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);

    layout->addWidget(createSettingsGroup());
    layout->addWidget(createButtonGroup());

    emit invalidateLayerWidgets();
    emit invalidateParameterWidgets();
}

void ToolboxWidget::setEditable(bool editable) {
    for (auto* widget : m_editables) {
        widget->setEnabled(editable);
    }
}

void ToolboxWidget::setCanvasColor(const QColor color) {
    m_canvasColor = color;
    emit invalidateParameterWidgets();
}

RenderMode ToolboxWidget::renderMode() const {
    return m_renderMode;
}

void ToolboxWidget::setRenderMode(RenderMode mode) {
    m_renderMode = mode;
    emit renderModeChanged(mode);
}

void ToolboxWidget::setMinIterationDuration(int minIterationDuration) {
    m_minIterationDuration = minIterationDuration;
    emit minIterationDurationChanged(m_minIterationDuration);
}

QWidget* ToolboxWidget::createSettingsGroup() {
    QScrollArea* settingsScrollArea = new QScrollArea(this);
    QWidget* settingsContainer = new QWidget(settingsScrollArea);

    settingsScrollArea->setWidgetResizable(true);
    settingsScrollArea->setFrameShape(QFrame::NoFrame);
    settingsScrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    settingsScrollArea->setWidget(settingsContainer);

    QVBoxLayout* settingsLayout = new QVBoxLayout(settingsContainer);
    settingsContainer->setLayout(settingsLayout);
    settingsLayout->setContentsMargins(0, 0, 2, 0);

    settingsLayout->addWidget(createExpandableWidget(settingsContainer,
        createCommonSettingsGroup(settingsContainer), "Common", style()->standardIcon(QStyle::SP_DialogResetButton)));
    settingsLayout->addWidget(createLayerSettingsGroups(settingsContainer));
    settingsLayout->addItem(new QSpacerItem(40, 20, QSizePolicy::Minimum, QSizePolicy::Expanding));

    return settingsScrollArea;
}

QWidget* ToolboxWidget::createCommonSettingsGroup(QWidget* parent) {
    QWidget* group = new QWidget(parent);

    QComboBox* stippleAlgorithmsBox = new QComboBox(group);
    stippleAlgorithmsBox->addItem("Linde-Buzo-Gray", QVariant::fromValue(StippleAlgorithm::LBG));
    stippleAlgorithmsBox->addItem("Coupled Linde-Buzo-Gray", QVariant::fromValue(StippleAlgorithm::CoupledLBG));
    stippleAlgorithmsBox->setToolTip("EXPERT: Choose an optimizing approach.\nNot couple means that all layers are optimized independently.");
    connect(stippleAlgorithmsBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
        [this, stippleAlgorithmsBox](int index) {
            auto options = m_stippler.options();
            options.stippleAlgorithm = stippleAlgorithmsBox->itemData(index).value<StippleAlgorithm>();
            m_stippler.setOptions(options);
        });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, [this, stippleAlgorithmsBox]() {
        stippleAlgorithmsBox->setCurrentIndex(static_cast<int>(m_stippler.options().stippleAlgorithm));
    });

    QComboBox* stippleModelBox = new QComboBox(group);
    stippleModelBox->addItem("Difference", QVariant::fromValue(StippleModel::Difference));
    stippleModelBox->addItem("Convolution (balanced)", QVariant::fromValue(StippleModel::Convolution));
    stippleModelBox->addItem("Convolution (filling)", QVariant::fromValue(StippleModel::ConvolutionFilling));
    stippleModelBox->addItem("Convolution (packing)", QVariant::fromValue(StippleModel::ConvolutionPacking));
    stippleModelBox->setToolTip("Choose how stipples are related to density.");
    connect(stippleModelBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
        [this, stippleModelBox](int index) {
            auto options = m_stippler.options();
            options.lbg.stippleModel = stippleModelBox->itemData(index).value<StippleModel>();
            m_stippler.setOptions(options);
        });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, [this, stippleModelBox]() {
        stippleModelBox->setCurrentIndex(static_cast<int>(m_stippler.options().lbg.stippleModel));
    });

    QComboBox* sizeModelBox = new QComboBox(group);
    sizeModelBox->addItem("Average", QVariant::fromValue(SizeModel::Average));
    sizeModelBox->addItem("Adjusted average", QVariant::fromValue(SizeModel::AdjustedAverage));
    sizeModelBox->setToolTip("EXPERT: Choose how stipples stipple size is estimated/determined.");
    connect(sizeModelBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
        [this, sizeModelBox](int index) {
            auto options = m_stippler.options();
            options.lbg.sizeModel = sizeModelBox->itemData(index).value<SizeModel>();
            m_stippler.setOptions(options);
        });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, [this, sizeModelBox]() {
        sizeModelBox->setCurrentIndex(static_cast<int>(m_stippler.options().lbg.sizeModel));
    });

    QComboBox* voronoiAlgorithmBox = new QComboBox(group);
    voronoiAlgorithmBox->addItem("Search (slow, exact)", QVariant::fromValue(VoronoiAlgorithm::Search));
    voronoiAlgorithmBox->addItem("Jump Flooding (fast, approximation)", QVariant::fromValue(VoronoiAlgorithm::JumpFlooding));
    voronoiAlgorithmBox->setToolTip("EXPERT: Choose an algorithm for computing the Voronoi diagram of layers.");
    connect(voronoiAlgorithmBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
        [this, voronoiAlgorithmBox](int index) {
            auto options = m_stippler.options();
            options.voronoiAlgorithm = voronoiAlgorithmBox->itemData(index).value<VoronoiAlgorithm>();
            m_stippler.setOptions(options);
        });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, [this, voronoiAlgorithmBox]() {
        voronoiAlgorithmBox->setCurrentIndex(static_cast<int>(m_stippler.options().voronoiAlgorithm));
    });

    QComboBox* hysteresisFunctionsBox = new QComboBox(group);
    hysteresisFunctionsBox->addItem("Constant", QVariant::fromValue(HysteresisFunction::Constant));
    hysteresisFunctionsBox->addItem("Linear", QVariant::fromValue(HysteresisFunction::Linear));
    hysteresisFunctionsBox->addItem("Linear (no MSE)", QVariant::fromValue(HysteresisFunction::LinearNoMSE));
    hysteresisFunctionsBox->addItem("Exponential (no MSE)", QVariant::fromValue(HysteresisFunction::ExponentialNoMSE));
    hysteresisFunctionsBox->setToolTip("EXPERT: Choose the influence of simulated annealing throughout iteration."
                                       "\nThat is how splitting and merging of stipples is dampened.");
    connect(hysteresisFunctionsBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
        [this, hysteresisFunctionsBox](int index) {
            auto options = m_stippler.options();
            options.lbg.hysteresisFunction = hysteresisFunctionsBox->itemData(index).value<HysteresisFunction>();
            m_stippler.setOptions(options);
        });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, [this, hysteresisFunctionsBox]() {
        hysteresisFunctionsBox->setCurrentIndex(static_cast<int>(m_stippler.options().lbg.hysteresisFunction));
    });

    QDoubleSpinBox* hysterresisStartBox = new QDoubleSpinBox(group);
    hysterresisStartBox->setRange(0.0, 3.0);
    hysterresisStartBox->setSingleStep(0.1);
    hysterresisStartBox->setToolTip("EXPERT: Choose an initial hysteresis value."
                                    "\nLower values means slower convergence and higher quality results.");
    connect(hysterresisStartBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        [this](double value) {
            auto options = m_stippler.options();
            options.lbg.hysteresisStart = value;
            m_stippler.setOptions(options);
        });
    connect(this, &ToolboxWidget::invalidateParameterWidgets,
        [this, hysterresisStartBox]() { hysterresisStartBox->setValue(m_stippler.options().lbg.hysteresisStart); });

    QDoubleSpinBox* hysteresisMaxBox = new QDoubleSpinBox(group);
    hysteresisMaxBox->setRange(0.0, 3.0);
    hysteresisMaxBox->setSingleStep(0.1);
    hysteresisMaxBox->setToolTip("EXPERT: Choose an expected hysteresis value when all iterations have passed."
                                 "\nHigher value means lower quality results.");
    connect(hysteresisMaxBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        [this](double value) {
            auto options = m_stippler.options();
            options.lbg.hysteresisMax = value;
            m_stippler.setOptions(options);
        });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, [this, hysteresisMaxBox]() {
        hysteresisMaxBox->setValue(m_stippler.options().lbg.hysteresisMax);
    });

    QSpinBox* maxIterationsBox = new QSpinBox(group);
    maxIterationsBox->setRange(1, 1000);
    maxIterationsBox->setToolTip("Choose after how many iterations the algorithm should stop (budget)."
                                 "\nMay stop earlier if appropriate.");
    connect(maxIterationsBox, QOverload<int>::of(&QSpinBox::valueChanged),
        [this](int value) {
            auto options = m_stippler.options();
            options.lbg.maxIterations = value;
            m_stippler.setOptions(options);
        });
    connect(this, &ToolboxWidget::invalidateParameterWidgets,
        [this, maxIterationsBox]() { maxIterationsBox->setValue(m_stippler.options().lbg.maxIterations); });

    QCheckBox* forcedCooldownBox = new QCheckBox(group);
    forcedCooldownBox->setToolTip("Enable to gradually supress splitting and removal during the last three iterations."
                                  "\nThis hides bad converge (ideal for underbudgeted max. iterations).");
    connect(forcedCooldownBox, QOverload<int>::of(&QCheckBox::stateChanged),
        [this](int state) {
            auto options = m_stippler.options();
            options.lbg.forcedCooldown = (state == Qt::Checked);
            m_stippler.setOptions(options);
        });
    connect(this, &ToolboxWidget::invalidateParameterWidgets,
        [this, forcedCooldownBox]() { forcedCooldownBox->setCheckState(m_stippler.options().lbg.forcedCooldown ? Qt::Checked : Qt::Unchecked); });

    QDoubleSpinBox* errorJitterPercentageBox = new QDoubleSpinBox(group);
    errorJitterPercentageBox->setRange(0.0, 1.0);
    errorJitterPercentageBox->setSingleStep(0.05);
    errorJitterPercentageBox->setDecimals(2);
    errorJitterPercentageBox->setToolTip("EXPERT: Choose how likely it is that a stipple must be 'perfect' during iteration."
                                         "\nA higher number can break 'chains of stipples' (Pareto fronts) but also impact convergence.");
    connect(errorJitterPercentageBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        [this](double value) {
            auto options = m_stippler.options();
            options.lbg.errorReductionChance = value;
            m_stippler.setOptions(options);
        });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, [this, errorJitterPercentageBox]() {
        errorJitterPercentageBox->setValue(m_stippler.options().lbg.errorReductionChance);
    });

    QDoubleSpinBox* splitJitterSpreadBox = new QDoubleSpinBox(group);
    splitJitterSpreadBox->setRange(0.0, 10.0);
    splitJitterSpreadBox->setSingleStep(0.1);
    splitJitterSpreadBox->setDecimals(2);
    splitJitterSpreadBox->setToolTip("EXPERT: Choose how randomness affects splitting of stipples (radius).");
    connect(splitJitterSpreadBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        [this](double value) {
            auto options = m_stippler.options();
            options.lbg.splitJitterSpread = value;
            m_stippler.setOptions(options);
        });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, [this, splitJitterSpreadBox]() {
        splitJitterSpreadBox->setValue(m_stippler.options().lbg.splitJitterSpread);
    });

    QSpinBox* voronoiScaleBox = new QSpinBox(group);
    voronoiScaleBox->setRange(25, 400);
    voronoiScaleBox->setSingleStep(25);
    voronoiScaleBox->setSuffix("%");
    voronoiScaleBox->setToolTip("EXPERT: Choose a discretization for the Voronoi diagram relative to image resolution."
                                "\nA higher number increases percision, but also slows down performance and vice versa.");
    connect(voronoiScaleBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int value) {
        auto oldValue = m_stippler.options().voronoiScale;
        auto options = m_stippler.options();
        options.voronoiScale = value / 100.0f;
        m_stippler.setOptions(options);
    });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, [this, voronoiScaleBox]() {
        voronoiScaleBox->setValue(m_stippler.options().voronoiScale * 100.0f);
    });
    connect(voronoiScaleBox, &QAbstractSpinBox::editingFinished, [this, voronoiScaleBox]() {
        double oldValue = m_stippler.options().voronoiScale;
        int value = voronoiScaleBox->value();
        int step = voronoiScaleBox->singleStep();
        if (value % step != 0) {
            voronoiScaleBox->setValue((value / step) * step);
        }
    });

    QSpinBox* borderWidthBox = new QSpinBox(group);
    borderWidthBox->setRange(0, 100);
    borderWidthBox->setToolTip(
        "Choose the size of a virtual border that should be added to the image.");
    connect(borderWidthBox, QOverload<int>::of(&QSpinBox::valueChanged),
        [this](int value) {
            auto options = m_stippler.options();
            options.borderWidth = value;
            m_stippler.setOptions(options);
        });
    connect(this, &ToolboxWidget::invalidateParameterWidgets,
        [this, borderWidthBox]() { borderWidthBox->setValue(m_stippler.options().borderWidth); });

    QPushButton* canvasColorButton = new QPushButton(group);
    canvasColorButton->setToolTip("Choose a color for the canvas (non-stipple background).");
    connect(canvasColorButton, &QPushButton::clicked, [this, canvasColorButton]() {
        Color color(m_canvasColor.red(), m_canvasColor.green(), m_canvasColor.blue(), 255);
        Color newColor = colorDialog(color);
        QPixmap pixmap = colorPixmap(newColor, canvasColorButton);
        canvasColorButton->setIcon(QIcon(pixmap));
        canvasColorButton->setIconSize(pixmap.size());
        m_canvasColor = QColor(newColor.r(), newColor.g(), newColor.b(), 255);
        emit canvasColorChanged(m_canvasColor);
    });
    connect(this, &ToolboxWidget::invalidateParameterWidgets,
        [this, canvasColorButton]() {
            Color color(m_canvasColor.red(), m_canvasColor.green(), m_canvasColor.blue(), 255);
            QPixmap pixmap = colorPixmap(color, QSize(180, 15));
            canvasColorButton->setIcon(QIcon(pixmap));
            canvasColorButton->setIconSize(pixmap.size());
        });

    QComboBox* renderModeBox = new QComboBox(group);
    renderModeBox->addItem("Stipples with background (density-based)", RenderMode::RasterStipplesWithBackground);
    renderModeBox->addItem("Stipples (density-based)", RenderMode::RasterStipples);
    renderModeBox->addItem("Background  (density-based)", RenderMode::RasterBackground);
    renderModeBox->addItem("Stipples (layer-based)", RenderMode::PainterStipples);
    renderModeBox->addItem("Highlighted stipples (layer-based)", RenderMode::PainterHighlightedStipples);
    renderModeBox->addItem("Density maps (layer-based)", RenderMode::PainterDensity);
    renderModeBox->setToolTip("Choose an approach for drawing.");
    connect(renderModeBox, QOverload<int>::of(&QComboBox::currentIndexChanged), [this, hysteresisFunctionsBox](int index) {
        m_renderMode = static_cast<RenderMode>(index);
        emit renderModeChanged(m_renderMode);
    });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, [this, renderModeBox]() {
        renderModeBox->setCurrentIndex(m_renderMode);
    });

    QSpinBox* minIterationDurationBox = new QSpinBox(group);
    minIterationDurationBox->setRange(0, 10000);
    minIterationDurationBox->setSingleStep(100);
    minIterationDurationBox->setSuffix("ms");
    minIterationDurationBox->setToolTip("Choose how long intermediate results should be visible (0 = disabled).");
    connect(minIterationDurationBox, QOverload<int>::of(&QSpinBox::valueChanged), [this](int value) {
        setMinIterationDuration(value);
    });
    connect(this, &ToolboxWidget::invalidateParameterWidgets,
        [this, minIterationDurationBox]() { minIterationDurationBox->setValue(m_minIterationDuration); });

    QFormLayout* layout = new QFormLayout(group);
    layout->setContentsMargins(2, 0, 2, 0);
    group->setLayout(layout);
    layout->addRow("Stippling algorithm:", stippleAlgorithmsBox);
    layout->addRow("Stipple model:", stippleModelBox);
    layout->addRow("Stipple size model:", sizeModelBox);
    layout->addRow("Voronoi algorithm:", voronoiAlgorithmBox);
    layout->addRow("Hysteresis function:", hysteresisFunctionsBox);
    layout->addRow("Hysteresis start:", hysterresisStartBox);
    layout->addRow("Hysteresis max.:", hysteresisMaxBox);
    layout->addRow("Maximum iterations:", maxIterationsBox);
    layout->addRow("Forced cooldown:", forcedCooldownBox);
    layout->addRow("Error-reduction:", errorJitterPercentageBox);
    layout->addRow("Split jitter:", splitJitterSpreadBox);
    layout->addRow("Voronoi scale:", voronoiScaleBox);
    layout->addRow("Border width:", borderWidthBox);
    layout->addRow("Canvas color:", canvasColorButton);
    layout->addRow("Renderer:", renderModeBox);
    layout->addRow("Iteration display:", minIterationDurationBox);

    m_editables.append(group);

    return group;
}

QWidget* ToolboxWidget::createExpandableLayerWidget(QWidget* parent, size_t index) {
    QWidget* container = new QWidget(parent);
    QVBoxLayout* containerLayout = new QVBoxLayout(container);
    container->setLayout(containerLayout);
    containerLayout->setContentsMargins(0, 0, 0, 0);

    auto layerName = [this, index]() -> QString {
        return "Layer " + QFileInfo(m_layers[index].imagePath).baseName();
    };
    auto layerIcon = [this, index]() -> QIcon {
        const QSize layerIconSize(15, 15);
        return QIcon(colorPixmap(m_layers[index].stippler.lbg.color, layerIconSize));
    };

    QWidget* bodyWidget = createLayerSettingsGroup(parent, index);
    ExpandableWidgetHeader* header = new ExpandableWidgetHeader(bodyWidget, layerName(), layerIcon(), container);
    connect(this, &ToolboxWidget::layerColorChanged, container, [this, header, layerIcon]() {
        header->updateIconLabel(layerIcon());
    });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, container, [this, header, layerName]() {
        header->updateTextLabel(layerName());
    });

    QWidget* arrowsContainer = new QWidget(parent);
    QVBoxLayout* arrowsContainerLayout = new QVBoxLayout(arrowsContainer);
    arrowsContainer->setLayout(arrowsContainerLayout);
    arrowsContainerLayout->setContentsMargins(2, 0, 2, 0);
    arrowsContainerLayout->setSpacing(0);

    QPushButton* buttonUp = new QPushButton(style()->standardIcon(QStyle::SP_TitleBarShadeButton), "", arrowsContainer);
    buttonUp->setFlat(true);
    buttonUp->setFixedSize(14, 14);
    if (index > 0) {
        connect(buttonUp, &QPushButton::pressed, [this, index]() {
            if (index > 0) {
                m_layers.swapItemsAt(index, index - 1);
                emit invalidateParameterWidgets();
                emit layerColorChanged();
                emit layerStyleChanged();
            }
        });
    } else {
        buttonUp->setDisabled(true);
    }
    arrowsContainerLayout->addWidget(buttonUp);

    QPushButton* buttonDown = new QPushButton(style()->standardIcon(QStyle::SP_TitleBarUnshadeButton), "", arrowsContainer);
    buttonDown->setFlat(true);
    buttonDown->setFixedSize(14, 14);
    if (index < (m_layers.count() - 1)) {
        connect(buttonDown, &QPushButton::pressed, [this, index]() {
            if (index < (m_layers.count() - 1)) {
                m_layers.swapItemsAt(index, index + 1);
                emit invalidateParameterWidgets();
                emit layerColorChanged();
                emit layerStyleChanged();
            }
        });
    } else {
        buttonDown->setDisabled(true);
    }
    arrowsContainerLayout->addWidget(buttonDown);

    QHBoxLayout* headerLayout = reinterpret_cast<QHBoxLayout*>(header->layout());
    headerLayout->addWidget(arrowsContainer);

    containerLayout->addWidget(header);
    containerLayout->addWidget(bodyWidget);

    return container;
}

QWidget* ToolboxWidget::createLayerSettingsGroups(QWidget* parent) {
    QWidget* containerWidget = new QWidget(parent);
    QVBoxLayout* containerLayout = new QVBoxLayout(containerWidget);
    containerLayout->setContentsMargins(0, 0, 0, 0);
    containerWidget->setLayout(containerLayout);
    connect(this, &ToolboxWidget::invalidateLayerWidgets, [this, containerWidget, containerLayout]() {
        for (auto& group : m_layerGroups) {
            containerLayout->removeWidget(group.get());
        }
        m_layerGroups.clear();
        for (int i = 0; i < m_layers.size(); ++i) {
            m_layerGroups.push_back(std::unique_ptr<QWidget>(createExpandableLayerWidget(containerWidget, i)));
            containerLayout->addWidget(m_layerGroups[i].get());
        }
        emit invalidateParameterWidgets();
    });

    m_editables.append(containerWidget);

    return containerWidget;
}

QWidget* ToolboxWidget::createLayerSettingsGroup(QWidget* parent, size_t layerIndex) {
    QWidget* group = new QWidget(parent);
    QFormLayout* layout = new QFormLayout(group);
    layout->setContentsMargins(2, 0, 2, 0);
    group->setLayout(layout);

    QPushButton* colorButton = new QPushButton(group);
    QPixmap pixmap = colorPixmap(m_layers[layerIndex].stippler.lbg.color, colorButton);
    colorButton->setIcon(QIcon(pixmap));
    colorButton->setIconSize(pixmap.size());
    colorButton->setToolTip("Choose a color.");
    connect(colorButton, &QPushButton::clicked, [this, colorButton, layerIndex]() {
        auto& layer = m_layers[layerIndex];
        layer.stippler.lbg.color = colorDialog(layer.stippler.lbg.color);
        QPixmap pixmap = colorPixmap(layer.stippler.lbg.color, colorButton);
        colorButton->setIcon(QIcon(pixmap));
        colorButton->setIconSize(pixmap.size());
        emit layerColorChanged();
        emit layerStyleChanged();
    });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, colorButton, [this, layerIndex, colorButton]() {
        auto& layer = m_layers[layerIndex];
        QPixmap pixmap = colorPixmap(layer.stippler.lbg.color, colorButton);
        colorButton->setIcon(QIcon(pixmap));
        colorButton->setIconSize(pixmap.size());
    });

    QCheckBox* showBox = new QCheckBox(group);
    showBox->setChecked(m_layers[layerIndex].visible);
    connect(showBox, &QCheckBox::clicked, [this, layerIndex](bool value) {
        auto& layer = m_layers[layerIndex];
        m_layers[layerIndex].visible = value;
        emit layerStyleChanged();
    });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, showBox, [this, layerIndex, showBox]() {
        showBox->setChecked(m_layers[layerIndex].visible);
    });

    QComboBox* stippleShapesBox = new QComboBox(group);
    stippleShapesBox->addItem("Circle", QVariant::fromValue(StippleShape::Circle));
    stippleShapesBox->addItem("Line", QVariant::fromValue(StippleShape::Line));
    stippleShapesBox->addItem("Rectangle", QVariant::fromValue(StippleShape::Rectangle));
    stippleShapesBox->addItem("Rhombus", QVariant::fromValue(StippleShape::Rhombus));
    stippleShapesBox->addItem("Ellipse", QVariant::fromValue(StippleShape::Ellipse));
    stippleShapesBox->addItem("Triangle", QVariant::fromValue(StippleShape::Triangle));
    stippleShapesBox->addItem("Rounded line", QVariant::fromValue(StippleShape::RoundedLine));
    stippleShapesBox->addItem("Rounded rectangle", QVariant::fromValue(StippleShape::RoundedRectangle));
    stippleShapesBox->addItem("Rounded rhombus", QVariant::fromValue(StippleShape::RoundedRhombus));
    stippleShapesBox->addItem("Rounded triangle", QVariant::fromValue(StippleShape::RoundedTriangle));
    stippleShapesBox->setToolTip("Choose a shape.");
    connect(stippleShapesBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
        [this, layerIndex, stippleShapesBox](int index) {
            m_layers[layerIndex].stippler.lbg.shape = stippleShapesBox->itemData(index).value<StippleShape>();
        });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, stippleShapesBox, [this, layerIndex, stippleShapesBox]() {
        stippleShapesBox->setCurrentIndex(static_cast<int>(m_layers[layerIndex].stippler.lbg.shape));
    });

    QLabel* shapeParameterLabel = new QLabel(group);
    shapeParameterLabel->setText("Aspect ratio:");
    QDoubleSpinBox* shapeParameterBox = new QDoubleSpinBox(group);
    shapeParameterBox->setRange(1.0, 50.0);
    shapeParameterBox->setSingleStep(0.1);
    shapeParameterBox->setToolTip(
        "Choose a width-to-height ratio for anisotropic shapes.");
    connect(shapeParameterBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        [this, layerIndex, stippleShapesBox](double value) {
            StippleShape currentShape = stippleShapesBox->itemData(stippleShapesBox->currentIndex()).value<StippleShape>();
            m_layers[layerIndex].stippler.lbg.shapeParameter = value;
        });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, shapeParameterBox, [this, layerIndex, shapeParameterBox]() {
        shapeParameterBox->setValue(m_layers[layerIndex].stippler.lbg.shapeParameter);
    });

    QDoubleSpinBox* shapeRadiusBox = new QDoubleSpinBox(group);
    shapeRadiusBox->setRange(0.01, 1.0);
    shapeRadiusBox->setSingleStep(0.1);
    shapeRadiusBox->setToolTip("Choose a radius for rounded shapes.");
    connect(shapeRadiusBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        [this, layerIndex](double value) { m_layers[layerIndex].stippler.lbg.shapeRadius = value; });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, shapeRadiusBox, [this, layerIndex, shapeRadiusBox]() {
        shapeRadiusBox->setValue(m_layers[layerIndex].stippler.lbg.shapeRadius);
    });

    auto updateShapeParameterWidgets = [this, layerIndex, stippleShapesBox, shapeParameterLabel, shapeParameterBox, shapeRadiusBox]() {
        StippleShape currentShape = stippleShapesBox->itemData(stippleShapesBox->currentIndex()).value<StippleShape>();
        shapeParameterBox->setValue(m_layers[layerIndex].stippler.lbg.shapeParameter);

        switch (currentShape) {
        case StippleShape::Circle:
        case StippleShape::Triangle:
        case StippleShape::RoundedTriangle:
            shapeParameterLabel->setText("Aspect ratio:");
            shapeParameterBox->setDisabled(true);
            break;
        case StippleShape::Line:
        case StippleShape::RoundedLine:
            shapeParameterLabel->setText("Line width:");
            shapeParameterBox->setRange(0.1, 50.0);
            shapeParameterBox->setDisabled(false);
            break;
        case StippleShape::Ellipse:
            shapeParameterLabel->setText("Aspect ratio:");
            shapeParameterBox->setRange(1.01, 50.0);
            shapeParameterBox->setDisabled(false);
            break;
        default:
            shapeParameterLabel->setText("Aspect ratio:");
            shapeParameterBox->setRange(1.0, 50.0);
            shapeParameterBox->setDisabled(false);
            break;
        }

        switch (currentShape) {
        case StippleShape::RoundedLine:
        case StippleShape::RoundedRectangle:
        case StippleShape::RoundedRhombus:
        case StippleShape::RoundedTriangle:
            shapeRadiusBox->setDisabled(false);
            break;
        default:
            shapeRadiusBox->setDisabled(true);
            break;
        }
    };
    connect(stippleShapesBox, QOverload<int>::of(&QComboBox::currentIndexChanged), updateShapeParameterWidgets);
    connect(this, &ToolboxWidget::invalidateParameterWidgets, shapeParameterBox, updateShapeParameterWidgets);

    QDoubleSpinBox* minSizeBox = new QDoubleSpinBox(group);
    minSizeBox->setRange(0.1, 100.0);
    minSizeBox->setSingleStep(0.5);
    minSizeBox->setToolTip(
        "Choose a minimum stipple size (diameter).");
    connect(minSizeBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        [this, layerIndex](double value) { m_layers[layerIndex].stippler.lbg.sizeMin = value; });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, minSizeBox, [this, layerIndex, minSizeBox]() {
        minSizeBox->setValue(m_layers[layerIndex].stippler.lbg.sizeMin);
    });

    QDoubleSpinBox* maxSizeBox = new QDoubleSpinBox(group);
    maxSizeBox->setRange(0.1, 100.0);
    maxSizeBox->setSingleStep(0.5);
    maxSizeBox->setToolTip(
        "Choose a maximum stipple size (diameter).");
    connect(maxSizeBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        [this, layerIndex](double value) { m_layers[layerIndex].stippler.lbg.sizeMax = value; });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, maxSizeBox, [this, layerIndex, maxSizeBox]() {
        maxSizeBox->setValue(m_layers[layerIndex].stippler.lbg.sizeMax);
    });

    QComboBox* sizeFunctionBox = new QComboBox(group);
    sizeFunctionBox->addItem("Linear", QVariant::fromValue(SizeFunction::Linear));
    sizeFunctionBox->addItem("Quadratic In", QVariant::fromValue(SizeFunction::QuadraticIn));
    sizeFunctionBox->addItem("Quadratic Out", QVariant::fromValue(SizeFunction::QuadraticOut));
    sizeFunctionBox->addItem("Quadratic InOut", QVariant::fromValue(SizeFunction::QuadraticInOut));
    sizeFunctionBox->addItem("Exponential In", QVariant::fromValue(SizeFunction::ExponentialIn));
    sizeFunctionBox->addItem("Exponential Out", QVariant::fromValue(SizeFunction::ExponentialOut));
    sizeFunctionBox->addItem("Exponential InOut", QVariant::fromValue(SizeFunction::ExponentialInOut));
    sizeFunctionBox->setToolTip("Choose how size is eased between minimum and maximum.");
    connect(sizeFunctionBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
        [this, layerIndex, sizeFunctionBox](int index) {
            m_layers[layerIndex].stippler.lbg.sizeFunction = sizeFunctionBox->itemData(index).value<SizeFunction>();
        });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, sizeFunctionBox, [this, layerIndex, sizeFunctionBox]() {
        sizeFunctionBox->setCurrentIndex(static_cast<int>(m_layers[layerIndex].stippler.lbg.sizeFunction));
    });

    QSpinBox* initialPointsBox = new QSpinBox(group);
    initialPointsBox->setRange(1, 100000);
    initialPointsBox->setToolTip(
        "EXPERT: Choose the amount of stipples the algorithm should start with.");
    connect(initialPointsBox, QOverload<int>::of(&QSpinBox::valueChanged),
        [this, layerIndex](int value) { m_layers[layerIndex].stippler.initialStipples = value; });
    connect(this, &ToolboxWidget::invalidateParameterWidgets, initialPointsBox,
        [this, layerIndex, initialPointsBox]() { initialPointsBox->setValue(m_layers[layerIndex].stippler.initialStipples); });

    layout->addRow("Show:", showBox);
    layout->addRow("Color:", colorButton);
    layout->addRow("Shape:", stippleShapesBox);
    layout->addRow(shapeParameterLabel, shapeParameterBox);
    layout->addRow("Rounding radius:", shapeRadiusBox);
    auto minSizeLabel = new ClickableLabel("Min. size:");
    connect(minSizeLabel, &ClickableLabel::doubleClicked, [this, minSizeBox]() {
        QApplication::beep();
        for (int l = 0; l < m_layers.size(); ++l) {
            m_layers[l].stippler.lbg.sizeMin = minSizeBox->value();
        }
        emit invalidateParameterWidgets();
    });
    layout->addRow(minSizeLabel, minSizeBox);
    auto maxSizeLabel = new ClickableLabel("Max. size:");
    connect(maxSizeLabel, &ClickableLabel::doubleClicked, [this, maxSizeBox]() {
        QApplication::beep();
        for (int l = 0; l < m_layers.size(); ++l) {
            m_layers[l].stippler.lbg.sizeMax = maxSizeBox->value();
        }
        emit invalidateParameterWidgets();
    });
    layout->addRow(maxSizeLabel, maxSizeBox);
    layout->addRow("Size mapping:", sizeFunctionBox);
    layout->addRow("Initial stipples:", initialPointsBox);

    return group;
}

QWidget* ToolboxWidget::createButtonGroup() {
    QWidget* group = new QWidget(this);

    QPushButton* stippleButton = new QPushButton("Stipple", this);
    stippleButton->setDefault(true);
    connect(stippleButton, &QPushButton::released, [this]() { emit start(); });

    auto settingsPath = [](const QLatin1String& key, std::function<QString(QString)> cb) {
        QSettings settings("settings.ini", QSettings::IniFormat);
        const QString path = cb(settings.value(key).toString());
        if (!path.isEmpty()) {
            qDebug() << path;
            settings.setValue(key, QFileInfo(path).absolutePath());
        }
    };

    auto getOpenImageNames = [this](const QString& defaultPath) {
        QString filter = tr("Image Files") + " (";
        QList<QByteArray> formats = QImageReader::supportedImageFormats();
        for (int i = 0; i < formats.count(); ++i) {
            if (i > 0)
                filter += " ";
            filter += QString("*.%1").arg(QString(formats[i]));
        }
        filter += ");;";

        return QFileDialog::getOpenFileNames(this, tr("Load Image(s)"), defaultPath, filter);
    };

    QPushButton* importButton = new QPushButton("Import", this);
    QMenu* importMenu = new QMenu(this);
    QAction* importAsDensityAction = importMenu->addAction("Density Layer(s)");
    QAction* importAsDualAction = importMenu->addAction("Black and White Decomposition");
    QMenu* importToyMenu = importMenu->addMenu("Procedural Layer(s)");
    QAction* importUniformAction = importToyMenu->addAction("Uniform Layers");
    QAction* importGrayscaleTestAction = importToyMenu->addAction("Grayscale Test Pattern");
    QAction* importInvertedGradientAction = importToyMenu->addAction("Inverted Gradient");
    QAction* importLinearGradientAction = importToyMenu->addAction("Linear Gradient");
    QAction* importGaussianFoveaSamplingAction = importToyMenu->addAction("Gaussian Fovea Sampling");
#if defined(QT_MULTIMEDIA_LIB)
    QMenu* cameraMenu = importMenu->addMenu("Camera");
    for (const auto& cameraDevice : QMediaDevices::videoInputs()) {
        QAction* cameraAction = cameraMenu->addAction(cameraDevice.description());
        cameraAction->setData(QVariant::fromValue(cameraDevice));
        connect(cameraAction, QOverload<bool>::of(&QAction::triggered), [this, cameraAction]() {
            const auto& cameraDevice = cameraAction->data().value<QCameraDevice>();
            auto* camera = new QCamera(cameraDevice);
            if (!camera->isAvailable()) {
                QMessageBox::warning(this, "Camera unavailable",
                    "The selected camera is unavailable or in use "
                    "by another application.");
                return;
            }
            emit importCamera(camera);
        });
    }
#endif
    importButton->setMenu(importMenu);
    connect(importAsDensityAction, QOverload<bool>::of(&QAction::triggered), [this, settingsPath, getOpenImageNames]() {
        settingsPath(QLatin1String("image_dir"), [&](auto defaultPath) {
            QStringList paths = getOpenImageNames(defaultPath);
            if (!paths.isEmpty()) {
                emit importAsDensity(paths);
                return paths.first();
            } else {
                return QString();
            }
        });
    });
    connect(importAsDualAction, QOverload<bool>::of(&QAction::triggered), [this, settingsPath, getOpenImageNames]() {
        settingsPath(QLatin1String("image_dir"), [&](auto defaultPath) {
            QStringList paths = getOpenImageNames(defaultPath);
            if (!paths.isEmpty()) {
                emit importAsDual(paths.first());
                return paths.first();
            } else {
                return QString();
            }
        });
    });
    connect(importUniformAction, QOverload<bool>::of(&QAction::triggered), [this]() {
        bool ok = true;
        auto num = QInputDialog::getInt(this, "Uniform", "Number of Layers", 1, 1, 10, 1, &ok, Qt::WindowFlags());
        auto density = QInputDialog::getDouble(this, "Uniform", "Density", 0.5, 0.0, 1.0, 2, &ok, Qt::WindowFlags(), 0.1);
        if (ok) {
            emit importLayers(std::move(uniformLayers(num, density)));
        }
    });
    connect(importGrayscaleTestAction, QOverload<bool>::of(&QAction::triggered), [this]() {
        emit importLayers(std::move(grayscaleTestLayers()));
    });
    connect(importInvertedGradientAction, QOverload<bool>::of(&QAction::triggered), [this]() {
        bool ok = true;
        auto cut = QMessageBox::question(this, "Inverted Gradient", "Show all transitions?",
            QMessageBox::Yes | QMessageBox::No);
        emit importLayers(std::move(invertedGradientLayers(cut == QMessageBox::Yes)));
    });
    connect(importLinearGradientAction, QOverload<bool>::of(&QAction::triggered), [this]() {
        emit importLayers(std::move(linearGradientLayer()));
    });
    connect(importGaussianFoveaSamplingAction, QOverload<bool>::of(&QAction::triggered), [this]() {
        auto options = m_stippler.options();
        options.voronoiScale = 1.0;
        m_stippler.setOptions(options);
        invalidateParameterWidgets();

#if 0
        // VISUS powerwall.
        QSize screenSizePx(10800, 4096);
        QSizeF screenSizeCm(582.66f, 220.9792f);
        float viewDistanceCm = 450.0f;
#else
        // Full HD monitor.
        QSize screenSizePx(1920, 1080);
        QSizeF screenSizeCm(53.35f, 30.1f);
        float viewDistanceCm = 60.0f;
#endif
        bool generate = false;

        QDialog* dialog = new QDialog(this);
        QFormLayout* layout = new QFormLayout();
        dialog->setWindowTitle("Set parameters");
        dialog->setWindowFlags(dialog->windowFlags().setFlag(Qt::WindowContextHelpButtonHint, false));
        dialog->setLayout(layout);

        QSpinBox* screenWidthPxBox = new QSpinBox(dialog);
        screenWidthPxBox->setRange(1, std::numeric_limits<short>::max());
        screenWidthPxBox->setValue(screenSizePx.width());
        connect(screenWidthPxBox, QOverload<int>::of(&QSpinBox::valueChanged),
            [&](int value) {
                screenSizePx.setWidth(value);
            });

        QSpinBox* screenHeightPxBox = new QSpinBox(dialog);
        screenHeightPxBox->setRange(1, std::numeric_limits<short>::max());
        screenHeightPxBox->setValue(screenSizePx.height());
        connect(screenHeightPxBox, QOverload<int>::of(&QSpinBox::valueChanged),
            [&](int value) {
                screenSizePx.setHeight(value);
            });

        QDoubleSpinBox* screenWidthCmBox = new QDoubleSpinBox(dialog);
        screenWidthCmBox->setRange(1.0, std::numeric_limits<short>::max());
        screenWidthCmBox->setDecimals(3);
        screenWidthCmBox->setValue(screenSizeCm.width());
        connect(screenWidthCmBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            [&](double value) {
                screenSizeCm.setWidth(value);
            });

        QDoubleSpinBox* screenHeightCmBox = new QDoubleSpinBox(dialog);
        screenHeightCmBox->setRange(1.0, std::numeric_limits<short>::max());
        screenHeightCmBox->setDecimals(3);
        screenHeightCmBox->setValue(screenSizeCm.height());
        connect(screenHeightCmBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            [&](double value) {
                screenSizeCm.setHeight(value);
            });

        QDoubleSpinBox* viewDistanceCmBox = new QDoubleSpinBox(dialog);
        viewDistanceCmBox->setRange(1.0, std::numeric_limits<short>::max());
        viewDistanceCmBox->setDecimals(3);
        viewDistanceCmBox->setValue(viewDistanceCm);
        connect(viewDistanceCmBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            [&](double value) {
                viewDistanceCm = static_cast<float>(value);
            });

        QPushButton* generateButton = new QPushButton("Generate", dialog);
        connect(generateButton, &QPushButton::pressed, [&]() {
            generate = true;
            dialog->close();
        });

        layout->addRow("Screen width (px)", screenWidthPxBox);
        layout->addRow("Screen height (px)", screenHeightPxBox);
        layout->addRow("Screen width (cm)", screenWidthCmBox);
        layout->addRow("Screen height (cm)", screenHeightCmBox);
        layout->addRow("Viewing distance (cm)", viewDistanceCmBox);
        layout->addWidget(generateButton);
        dialog->exec();

        if (generate) {
            auto [layers, sigmaX, sigmaY] = gaussianFoveaSamplingLayer(screenSizePx, screenSizeCm, viewDistanceCm);
            qInfo() << "Sigma (px)" << sigmaX << "," << sigmaY;
            qInfo() << "Screen size (px)" << screenSizePx.width() << "," << screenSizePx.height();
            qInfo() << "Screen size (cm)" << screenSizeCm.width() << "," << screenSizeCm.height();
            qInfo() << "Viewing distance (cm)" << viewDistanceCm;
            emit importLayers(std::move(layers));
        }
    });

    QPushButton* loadProjectButton = new QPushButton("Load Project", this);
    connect(loadProjectButton, &QPushButton::pressed, [this, settingsPath]() {
        settingsPath(QLatin1String("project_dir"), [this](auto defaultPath) {
            QString path = QFileDialog::getOpenFileName(this, tr("Load Project"), defaultPath,
                tr("Project File (*.json)"));
            if (!path.isEmpty()) {
                emit loadProject(path);
                emit invalidateParameterWidgets();
            }
            return path;
        });
    });

    QPushButton* saveProjectButton = new QPushButton("Save Project", this);
    connect(saveProjectButton, &QPushButton::pressed, [this, settingsPath]() {
        settingsPath(QLatin1String("project_dir"), [this](auto defaultPath) {
            QString path = QFileDialog::getSaveFileName(this, tr("Save Project"), defaultPath,
                tr("Project File (*.json)"));
            if (!path.isEmpty()) {
                emit saveProject(path);
            }
            return path;
        });
    });

    QPushButton* exportButton = new QPushButton("Export", this);
    QMenu* exportMenu = new QMenu(this);
    QAction* exportImageAction = exportMenu->addAction("Image");
    QAction* exportNaturalNeighborDataAction = exportMenu->addAction("Natural Neighbor Data (experimental)");
    exportButton->setMenu(exportMenu);
    connect(exportImageAction, QOverload<bool>::of(&QAction::triggered), [this]() {
        QString filter = tr("Bitmap Image") + " (";
        QList<QByteArray> formats = QImageWriter::supportedImageFormats();
        for (int i = 0; i < formats.count(); ++i) {
            if (i > 0)
                filter += " ";
            filter += QString("*.%1").arg(QString(formats[i]));
        }
        filter += ");;Scalable Vector Graphics (*.svg);;Portable Document Format (*.pdf)";

        QFileDialog dialog(this, tr("Export as"), QString(), filter);
        dialog.setAcceptMode(QFileDialog::AcceptSave);
        dialog.setDefaultSuffix("png");
        if (dialog.exec() == 0)
            return;

        QString path = dialog.selectedFiles().first();
        if (path.isEmpty())
            return;

        emit exportImage(path);
    });
    connect(exportNaturalNeighborDataAction, QOverload<bool>::of(&QAction::triggered), [this]() {
        emit exportNaturalNeighborData();
    });

    QGridLayout* layout = new QGridLayout(group);
    layout->setContentsMargins(0, 0, 2, 2);
    group->setLayout(layout);
    layout->addWidget(stippleButton, 0, 0, 1, 2);
    layout->addWidget(loadProjectButton, 1, 0);
    layout->addWidget(saveProjectButton, 1, 1);
    layout->addWidget(importButton, 2, 0);
    layout->addWidget(exportButton, 2, 1);

    m_editables.append(group);

    return group;
}
