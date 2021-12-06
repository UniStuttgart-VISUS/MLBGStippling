#include "clickablelabel.h"

ClickableLabel::ClickableLabel(const QString& text, QWidget* parent, Qt::WindowFlags flags)
    : QLabel(text, parent) {
}

void ClickableLabel::mousePressEvent(QMouseEvent* event) {
    emit clicked();
}

void ClickableLabel::mouseDoubleClickEvent(QMouseEvent* event) {
    emit doubleClicked();
}