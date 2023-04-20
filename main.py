from covertype_class import CovertypeClassifier


def main():
    clf = CovertypeClassifier()
    clf.load_data()
    clf.balance_data(Undersample=True)
    clf.train_val_test_split()
    clf.scale_data()

    clf.simple_heuristic_classification(clf.X_train)
    clf.evaluate(clf.y_pred_heuristic_test)

    clf.train_logistic_regression_classifier()
    clf.evaluate(clf.lr_pred)

    clf.train_decision_tree_classifier()
    clf.evaluate(clf.dtc_pred)

    clf.grid_search()
    clf.nn(with_grid_search=False)
    clf.evaluate(clf.y_pred_ann)



if __name__ == "__main__":
    main()