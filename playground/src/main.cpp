#include "layertoy.h"
#include "logger.h"
#include "mainwindow.h"
#include <QApplication>
#include <QtCore>

void measure(MainWindow& window, int samples, std::function<void(int, StipplerOptions&, Layers&)> next) {
    // Prepare.
    window.setCanvasColor(QColor("#ffffff"));
    window.setIterationCallback([](const auto& layerStipples, const auto& lindeBuzoGrayResult) {
        logf("  { \"done\": %d, \"iteration\": %d, "
             "\"stipples\": %d, \"splits\": %d, \"removals\": %d, \"merges\": %d, \"moves\": %d, "
             "\"totalDensityMeanSquaredError\": %f, "
             "\"totalIntersectionError\": %f, \"totalCellExcess\": %f, \"totalStippleExcess\": %f }\n",
            lindeBuzoGrayResult.done, lindeBuzoGrayResult.iteration, lindeBuzoGrayResult.total,
            lindeBuzoGrayResult.splits, lindeBuzoGrayResult.removals, lindeBuzoGrayResult.merges, lindeBuzoGrayResult.moves,
            lindeBuzoGrayResult.totalDensityMeanSquaredError, lindeBuzoGrayResult.totalIntersectionError,
            lindeBuzoGrayResult.totalCellExcess, lindeBuzoGrayResult.totalStippleExcess);
        if (!lindeBuzoGrayResult.done) {
            logf(",");
        }
    });
    window.setVisible(false);

    StipplerOptions options;
    Layers layers;
    quint64 totalTime = 0;
    QElapsedTimer timer;
    logf("[");
    for (int s = 0; s < samples; ++s) {
        // Sample.
        timer.start();
        next(s, options, layers);
        window.setOptions(options);
        window.setLayers(layers);

        qInfo("Sample %d (%d%%)", s, (s * 100) / samples);

        logf("{\n");
        QJsonDocument jsonParameters(window.parametersToJson());
        logf("\"sample\": %d,\n", s + 1);
        logf("\"imagePath\": \"%s\",\n", qPrintable(layers[0].imagePath));
        logf("\"imageWidth\": %i,\n", layers[0].image.width());
        logf("\"imageHeight\": %i,\n", layers[0].image.height());
        logf("\"parameters\": %s,\n", qPrintable(jsonParameters.toJson()));
        logf("\"iterations\": [\n");

        // Run.
        qint64 executionDuration = window.stipple();
        window.saveImage(QString("s%1.png").arg(s + 1));

        logf("],\n");
        logf("\"executionDuration\": %i\n", static_cast<int>(executionDuration));
        logf("}\n");
        if (s < samples - 1) {
            logf(",");
        }

        quint64 elapsed = timer.elapsed();
        totalTime += elapsed;
        quint64 timeLeft = (totalTime / (s + 1)) * (samples - (s + 1));
        qInfo("%5dms real %5dms stipple %-14s %10.2fmin total %-14s %10.2fmin left",
            (int)elapsed, (int)executionDuration, " ",
            totalTime / (1000.0f * 60.0f), " ", timeLeft / (1000.0f * 60.0f));
    }
    logf("]");
    logflush();

    window.setVisible(true);
}

void gradientExperiment(MainWindow& window) {
    const auto outerLayers = linearGradientLayer();

    std::seed_seq seq { 1, 2, 3, 4, 5 };
    std::mt19937 gen(seq);

    std::uniform_int_distribution<int> hysteresisFunctionD(0, 2);
    std::uniform_int_distribution<int> shapeD(0, 9);
    std::uniform_real_distribution<float> shapeParameterD(1.0f, 3.0f);
    std::uniform_real_distribution<float> shapeRadiusD(0.25f, 1.0f);
    std::uniform_real_distribution<float> sizeD(1.0f, 15.0f);
    std::uniform_int_distribution<int> sizeFunctionD(0, 6);
    std::uniform_real_distribution<float> stippleExcessMaxD(0.3f, 1.3f);
    std::uniform_int_distribution<int> initialStipplesD(1, 1000);

    const int samplesPerVariable = 1000;
    const int samples = 8 * samplesPerVariable;

    measure(window, samples, [&](const int sample, StipplerOptions& options, Layers& layers) {
        options.lbg.stippleModel = static_cast<StippleModel>(sample % 3);
        if (sample % 3 != 0) {
            // Re-run same paramters using other models.
            return;
        }
        // Global
        options.stippleAlgorithm = StippleAlgorithm::LBG; // Stays fixed (single-layer does not require coupling)
        options.voronoiAlgorithm = VoronoiAlgorithm::JumpFlooding; // Stays fixed for performance reasons.
        options.voronoiScale = 1.0f; // Stays fixed to avoid influence.
        options.borderWidth = 0.0f; // Stays fixed to avoid influence.
        options.lbg.hysteresisStart = 0.01f; // Stays fixed to avoid influence.
        options.lbg.hysteresisMax = 1.0f; // Stays fixed to avoid influence.
        options.lbg.hysteresisFunction = static_cast<HysteresisFunction>(hysteresisFunctionD(gen));
        options.lbg.maxIterations = 50; // Stays fixed to avoid influence.
        options.lbg.forcedCooldown = false; // Stays fixed to avoid influence.
        options.lbg.errorReductionChance = 0.0f; // Stays fixed to avoid influence.
        options.lbg.splitJitterSpread = 0.0f; // Stays fixed to avoid influence.

        // Layer
        layers = outerLayers;
        StipplerLayer& layer = layers[0].stippler;
        layer.lbg.shape = static_cast<StippleShape>(shapeD(gen));
        layer.lbg.shapeParameter = shapeParameterD(gen);
        layer.lbg.shapeRadius = shapeRadiusD(gen);
        auto sizeA = sizeD(gen);
        auto sizeB = sizeD(gen);
        layer.lbg.sizeMin = std::min(sizeA, sizeB);
        layer.lbg.sizeMax = std::max(sizeA, sizeB);
        layer.lbg.sizeFunction = static_cast<SizeFunction>(sizeFunctionD(gen));
        layer.initialStipples = initialStipplesD(gen);
    });
}

void uniformExperiment(MainWindow& window) {
    std::seed_seq seq { 1, 2, 3, 4, 5 };
    std::mt19937 gen(seq);
    std::uniform_int_distribution<int> layerD(1, 10);
    std::uniform_int_distribution<int> widthD(128, 1920);
    std::uniform_int_distribution<int> heightD(128, 1080);
    std::uniform_real_distribution<float> totalDensityD(0.0f, 1.0f);

    const int samplesPerVariable = 50;
    const int samples = 4 * samplesPerVariable;

    measure(window, samples, [&](const int sample, StipplerOptions& options, Layers& layers) {
        // Global
        options.stippleAlgorithm = StippleAlgorithm::CoupledLBG; // Stays fixed (multi-layer experiment)
        options.voronoiAlgorithm = VoronoiAlgorithm::JumpFlooding; // Stays fixed for performance reasons.
        options.voronoiScale = 1.0f; // Stays fixed to avoid influence.
        options.borderWidth = 0.0f; // Stays fixed to avoid influence.
        options.lbg.stippleModel = StippleModel::Convolution; // Stays fixed to avoid influence.
        options.lbg.sizeModel = SizeModel::AdjustedAverage; // Stays fixed to avoid influence.
        options.lbg.hysteresisStart = 0.01f; // Stays fixed to avoid influence.
        options.lbg.hysteresisMax = 1.0f; // Stays fixed to avoid influence.
        options.lbg.hysteresisFunction = HysteresisFunction::LinearNoMSE; // Stays fixed to avoid influence.
        options.lbg.maxIterations = 50; // Stays fixed to avoid influence.
        options.lbg.forcedCooldown = false; // Stays fixed to avoid influence.
        options.lbg.errorReductionChance = 0.0f; // Stays fixed to avoid influence.
        options.lbg.splitJitterSpread = 0.0f; // Stays fixed to avoid influence.

        const int MaxPixels = 4096 * 4096 * 3; // Estimate of upper memory bound.
        int layerCount, width, height;
        do {
            layerCount = layerD(gen);
            width = widthD(gen);
            height = heightD(gen);

        } while (layerCount * width * height > MaxPixels);
        /*layers = uniformLayers(layerCount, width, height, totalDensityD(gen));
        for (int i = 0; i < layerCount; ++i) {
            layers[i].stippler.initialStipples = 1; // Stays fixed to avoid influence.
            layers[i].stippler.lbg.shape = StippleShape::Circle; // Stays fixed to avoid influence.
            layers[i].stippler.lbg.sizeMin = 2.0f; // Stays fixed to avoid influence.
            layers[i].stippler.lbg.sizeMax = 2.0f; // Stays fixed to avoid influence.
            layers[i].stippler.lbg.sizeFunction = SizeFunction::Linear; // Stays fixed to avoid influence.
            layers[i].stippler.initialStipples = 100; // Stays fixed to avoid influence.
        }*/
    });
}

int main(int argc, char* argv[]) {
    registerConverters();

    QLocale::setDefault(QLocale("en_US"));

    QApplication app(argc, argv);
    app.setApplicationName("MLBG Stippling Playground");

    QCommandLineParser parser;
    parser.addPositionalArgument("images", "Image files.");
    QCommandLineOption experiment("experiment", "Run experiment.", "name");
    parser.addOption(experiment);
    QCommandLineOption watchOption("watch", "Watches a directory.", "directory");
    parser.addOption(watchOption);
    parser.process(app);

    MainWindow window(std::move(grayscaleTestLayers()));

    QTimer watcherThrottle;
    QFileSystemWatcher watcher;

    const QStringList positionalArguments = parser.positionalArguments();
    if (parser.isSet(watchOption)) {
        QDir dir(parser.value(watchOption));
        if (!dir.exists()) {
            qCritical() << dir.path() << " must be a directory";
            return -1;
        }
        auto throttledReload = [dir, &window]() {
            QDirIterator it(dir.path(), { "*.png" }, QDir::Files);
            QStringList paths;
            while (it.hasNext()) {
                paths << it.next();
            }
            window.loadImages(paths);
            window.stipple();
        };
        watcherThrottle.setSingleShot(true);
        QObject::connect(&watcherThrottle, &QTimer::timeout, throttledReload);
        watcher.addPath(dir.path());
        auto throttleTrigger = [&watcherThrottle](const QString& path) {
            watcherThrottle.start(1000);
        };
        QObject::connect(&watcher, &QFileSystemWatcher::fileChanged, throttleTrigger);
        QObject::connect(&watcher, &QFileSystemWatcher::directoryChanged, throttleTrigger);
    } else if (!positionalArguments.isEmpty()) {
        auto first = positionalArguments.first();
        if (first.endsWith(".json")) {
            window.loadProject(first);
        } else {
            window.loadImages(positionalArguments);
        }
    }

    window.show();
    if (parser.isSet(watchOption)) {
        watcherThrottle.start(0);
    } else if (parser.isSet(experiment)) {
        if (parser.value(experiment) == QLatin1String("gradient")) {
            gradientExperiment(window);
            return 0;
        } else if (parser.value(experiment) == QLatin1String("uniform")) {
            uniformExperiment(window);
            return 0;
        } else {
            qCritical() << "Unknown experiment";
            return -1;
        }
    }

    return app.exec();
}
