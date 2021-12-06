#ifndef CLICKABLELABEL_H
#define CLICKABLELABEL_H

#include <QLabel>
#include <QWidget>
#include <Qt>

class ClickableLabel : public QLabel {
    Q_OBJECT
public:
    explicit ClickableLabel(const QString& text, QWidget* parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags());
    virtual ~ClickableLabel() = default;

signals:
    void clicked();
    void doubleClicked();

protected:
    void mousePressEvent(QMouseEvent* event);
    void mouseDoubleClickEvent(QMouseEvent* event);
};

#endif // CLICKABLELABEL_H
