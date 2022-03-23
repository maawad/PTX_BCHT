#pragma once
#include <qstring.h>
#include <qsyntaxhighlighter.h>

struct HighlightingRule {
  QRegularExpression pattern;
  QTextCharFormat format;
};

class ptx_highlighter : public QSyntaxHighlighter {
 public:
  ptx_highlighter(QTextDocument *parent = 0) : QSyntaxHighlighter(parent) {
    HighlightingRule rule;
    QTextCharFormat instructions_format;
    instructions_format.setForeground(Qt::darkGreen);
    instructions_format.setFontWeight(QFont::Bold);
    const QString instructions[] = {QStringLiteral("\\badd\\b"),     QStringLiteral("\\bsub\\b"),
                                    QStringLiteral("\\bld\\b"),      QStringLiteral("\\bst\\b"),
                                    QStringLiteral("\\batom\\b"),    QStringLiteral("\\bmad\\b"),
                                    QStringLiteral("\\bmov\\b"),     QStringLiteral("\\bbfind\\b"),
                                    QStringLiteral("\\bset\\b"),     QStringLiteral("\\bpopc\\b"),
                                    QStringLiteral("\\bshfl\\b"),    QStringLiteral("\\bsync\\b"),
                                    QStringLiteral("\\bbra\\b"),     QStringLiteral("\\buni\\b"),
                                    QStringLiteral("\\bvote\\b"),    QStringLiteral("\\bballot\\b"),
                                    QStringLiteral("\\bsetp\\b"),    QStringLiteral("\\bne\\b"),
                                    QStringLiteral("\\beq\\b"),      QStringLiteral("\\bidx\\b"),
                                    QStringLiteral("\\brelaxed\\b"), QStringLiteral("\\bgpu\\b"),
                                    QStringLiteral("\\bmul\\b"),     QStringLiteral("\\bwide\\b"),
                                    QStringLiteral("\\bparam\\b"),   QStringLiteral("\\breg\\b"),
                                    QStringLiteral("\\bxor\\b"),     QStringLiteral("\\bshl\\b"),
                                    QStringLiteral("\\bshr\\b"),     QStringLiteral("\\brem\\b"),
                                    QStringLiteral("\\bglobal\\b")};

    for (const QString &pattern : instructions) {
      rule.pattern = QRegularExpression(pattern);
      rule.format  = instructions_format;
      highlightingRules.append(rule);
    }
    for (const QString &pattern : instructions) {
      rule.pattern = QRegularExpression(pattern);
      rule.format  = instructions_format;
      highlightingRules.append(rule);
    }

    QTextCharFormat types_format;
    types_format.setForeground(Qt::cyan);
    types_format.setFontWeight(QFont::Bold);
    const QString types[] = {QStringLiteral("\\bs8\\b"),
                             QStringLiteral("\\bs16\\b"),
                             QStringLiteral("\\bs32\\b"),
                             QStringLiteral("\\bs64\\b"),
                             QStringLiteral("\\bu8\\b"),
                             QStringLiteral("\\bu16\\b"),
                             QStringLiteral("\\bu32\\b"),
                             QStringLiteral("\\bu64\\b"),
                             QStringLiteral("\\bb8\\b"),
                             QStringLiteral("\\bb16\\b"),
                             QStringLiteral("\\bb32\\b"),
                             QStringLiteral("\\bb64\\b")};

    for (const QString &pattern : types) {
      rule.pattern = QRegularExpression(pattern);
      rule.format  = types_format;
      highlightingRules.append(rule);
    }

    QTextCharFormat comment;
    comment.setForeground(Qt::darkYellow);
    rule.pattern = QRegularExpression(QStringLiteral("//[^\n]*"));
    rule.format  = comment;
    highlightingRules.append(rule);
  }

  void ptx_highlighter::highlightBlock(const QString &text) {
    for (const HighlightingRule &rule : qAsConst(highlightingRules)) {
      QRegularExpressionMatchIterator matchIterator = rule.pattern.globalMatch(text);
      while (matchIterator.hasNext()) {
        QRegularExpressionMatch match = matchIterator.next();
        setFormat(match.capturedStart(), match.capturedLength(), rule.format);
      }
    }
  }

 private:
  QVector<HighlightingRule> highlightingRules;
};