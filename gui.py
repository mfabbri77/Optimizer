import sys
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QTextEdit, QLabel, QComboBox, 
                             QProgressBar, QStatusBar, QTabWidget, QSplitter, QCheckBox)
from PyQt5.QtGui import QTextOption
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from data_handling import read_data
from parametric_functions import (LinearFunction, QuadraticFunction, ExponentialFunction, GaussianFunction,
                                 SineFunction, LogarithmicFunction, PowerLawFunction, PolynomialFunction,
                                 LogisticFunction, HyperbolicTangentFunction)
from optimization import select_best_function, advanced_optimization
from visualization import plot_results, plot_residuals


class OptimizationThread(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    time_updated = pyqtSignal(float, float)
    optimization_complete = pyqtSignal(object, object)

    def __init__(self, data, function, fast_mode, functions):
        super().__init__()
        self.data = data
        self.function = function
        self.fast_mode = fast_mode
        self.functions = functions
        self.start_time = None

    def run(self):
        self.start_time = time.time()
        x, y = self.data[:2]
        if self.function == "Auto Select":
            self.status_updated.emit("Selecting best function...")
            best_func = select_best_function(x, y, self.functions, 
                                             progress_callback=self.progress_updated.emit,
                                             status_callback=self.status_updated.emit,
                                             time_callback=self.update_time)
        else:
            function_classes = {
                "Linear": LinearFunction,
                "Quadratic": QuadraticFunction,
                "Exponential": ExponentialFunction,
                "Gaussian": GaussianFunction,
                "Sine": SineFunction,
                "Logarithmic": LogarithmicFunction,
                "Power Law": PowerLawFunction,
                "Polynomial (degree 2)": lambda: PolynomialFunction(degree=2),
                "Polynomial (degree 3)": lambda: PolynomialFunction(degree=3),
                "Polynomial (degree 4)": lambda: PolynomialFunction(degree=4),
                "Polynomial (degree 5)": lambda: PolynomialFunction(degree=5),
                "Logistic": LogisticFunction,
                "Hyperbolic Tangent": HyperbolicTangentFunction
            }
            best_func = function_classes[self.function]()
            self.progress_updated.emit(50)
            self.status_updated.emit(f"Selected {best_func.__class__.__name__} function")
        
        self.status_updated.emit("Starting parameter optimization...")
        optimized_params = advanced_optimization(best_func, x, y, 
                                                 progress_callback=self.progress_updated.emit,
                                                 status_callback=self.status_updated.emit,
                                                 time_callback=self.update_time,
                                                 fast_mode=self.fast_mode)
        self.status_updated.emit("Optimization completed")
        self.optimization_complete.emit(best_func, optimized_params)

    def update_time(self, progress):
        elapsed_time = time.time() - self.start_time
        estimated_total_time = elapsed_time / (progress / 100)
        remaining_time = estimated_total_time - elapsed_time
        self.time_updated.emit(elapsed_time, remaining_time)

class OptimizationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parameter Optimization GUI")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Creiamo un QSplitter per dividere il pannello sinistro e destro
        self.splitter = QSplitter(Qt.Horizontal)
        self.layout.addWidget(self.splitter)

        # Creiamo e aggiungiamo il pannello sinistro
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.setup_left_panel()
        self.splitter.addWidget(self.left_panel)

        # Creiamo e aggiungiamo il pannello destro
        self.right_panel = QTabWidget()
        self.setup_right_panel()
        self.splitter.addWidget(self.right_panel)

        # Impostiamo le dimensioni iniziali dei pannelli
        self.splitter.setSizes([300, 900])  # Larghezza iniziale del pannello sinistro: 300px

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.elapsed_time_label = QLabel()
        self.remaining_time_label = QLabel()
        self.status_bar.addPermanentWidget(self.elapsed_time_label)
        self.status_bar.addPermanentWidget(self.remaining_time_label)

        self.data = None
        self.best_func = None
        self.optimized_params = None

    def setup_left_panel(self):
        self.file_button = QPushButton("Select Data File")
        self.file_button.clicked.connect(self.load_data)
        self.left_layout.addWidget(self.file_button)

        self.function_combo = QComboBox()
        self.function_combo.addItems(["Auto Select", "Linear", "Quadratic", "Exponential", "Gaussian",
                                      "Sine", "Logarithmic", "Power Law", 
                                      "Polynomial (degree 2)", "Polynomial (degree 3)", 
                                      "Polynomial (degree 4)", "Polynomial (degree 5)",
                                      "Logistic", "Hyperbolic Tangent"])
        self.left_layout.addWidget(self.function_combo)

        self.fast_mode_checkbox = QCheckBox("Fast Mode (Less Accurate)")
        self.left_layout.addWidget(self.fast_mode_checkbox)

        self.optimize_button = QPushButton("Optimize Parameters")
        self.optimize_button.clicked.connect(self.run_optimization)
        self.left_layout.addWidget(self.optimize_button)

        self.progress_bar = QProgressBar()
        self.left_layout.addWidget(self.progress_bar)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        # Imposta il QTextEdit per non troncare le righe
        self.result_text.setLineWrapMode(QTextEdit.NoWrap)
        
        # Abilita lo scorrimento orizzontale
        self.result_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Imposta l'opzione di testo per non troncare le parole
        text_option = QTextOption()
        text_option.setWrapMode(QTextOption.NoWrap)
        self.result_text.document().setDefaultTextOption(text_option)
        
        self.left_layout.addWidget(self.result_text)

        # Aggiungiamo uno stretcher per spingere tutti i widget verso l'alto
        self.left_layout.addStretch(1)

    def setup_right_panel(self):
        # Results plot tab
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.right_panel.addTab(self.canvas, "Results Plot")

        # Residuals plot tab
        self.residual_figure = Figure(figsize=(5, 4), dpi=100)
        self.residual_canvas = FigureCanvas(self.residual_figure)
        self.right_panel.addTab(self.residual_canvas, "Residuals Plot")

    def load_data(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "Text Files (*.txt)")
        if filename:
            try:
                self.data = read_data(filename)
                self.result_text.append(f"Data loaded from {filename}")
                self.result_text.append(f"Data shape: {len(self.data[0])} points, {len(self.data)} dimensions")
                self.status_bar.showMessage("Data loaded successfully")
            except Exception as e:
                self.result_text.append(f"Error loading data: {str(e)}")
                self.status_bar.showMessage("Error loading data")

    def run_optimization(self):
        if self.data is None:
            self.result_text.append("Please load data first.")
            self.status_bar.showMessage("No data loaded")
            return

        self.optimize_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Initializing optimization process...")

        selected_function = self.function_combo.currentText()
        fast_mode = self.fast_mode_checkbox.isChecked()

        functions = [
            LinearFunction(), 
            QuadraticFunction(), 
            ExponentialFunction(), 
            GaussianFunction(),
            SineFunction(), 
            LogarithmicFunction(), 
            PowerLawFunction(), 
            PolynomialFunction(degree=2),
            PolynomialFunction(degree=3),
            PolynomialFunction(degree=4),
            PolynomialFunction(degree=5),
            LogisticFunction(),
            HyperbolicTangentFunction()
        ]

        self.optimization_thread = OptimizationThread(self.data, selected_function, fast_mode, functions)
        self.optimization_thread.progress_updated.connect(self.update_progress)
        self.optimization_thread.status_updated.connect(self.update_status)
        self.optimization_thread.time_updated.connect(self.update_time)
        self.optimization_thread.optimization_complete.connect(self.optimization_finished)
        self.optimization_thread.start()

    def update_status(self, message):
        self.status_bar.showMessage(message)
        self.result_text.append(message)

    def update_time(self, elapsed_time, remaining_time):
        elapsed_str = self.format_time(elapsed_time)
        remaining_str = self.format_time(remaining_time)
        self.elapsed_time_label.setText(f"Elapsed: {elapsed_str}")
        self.remaining_time_label.setText(f"Remaining: {remaining_str}")

    def format_time(self, seconds):
        hours, rem = divmod(seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def optimization_finished(self, best_func, optimized_params):
        self.best_func = best_func
        self.optimized_params = optimized_params

        func_str = self.get_function_string(best_func, optimized_params)
        self.result_text.append(f"Best function: {best_func.__class__.__name__}")
        self.result_text.append(f"Optimized function: {func_str}")
        self.result_text.append(f"Optimized parameters: {optimized_params}")

        self.plot_results()
        self.optimize_button.setEnabled(True)
        self.status_bar.showMessage("Optimization completed")

    def get_function_string(self, func, params):
        if isinstance(func, LinearFunction):
            return f"f(x) = {params[0]:.4f}x + {params[1]:.4f}"
        elif isinstance(func, QuadraticFunction):
            return f"f(x) = {params[0]:.4f}x^2 + {params[1]:.4f}x + {params[2]:.4f}"
        elif isinstance(func, ExponentialFunction):
            return f"f(x) = {params[0]:.4f} * exp({params[1]:.4f}x) + {params[2]:.4f}"
        elif isinstance(func, GaussianFunction):
            return f"f(x) = {params[0]:.4f} * exp(-((x - {params[1]:.4f})^2) / (2 * {params[2]:.4f}^2))"
        elif isinstance(func, SineFunction):
            return f"f(x) = {params[0]:.4f} * sin({params[1]:.4f}x + {params[2]:.4f}) + {params[3]:.4f}"
        elif isinstance(func, LogarithmicFunction):
            return f"f(x) = {params[0]:.4f} * log({params[1]:.4f}x) + {params[2]:.4f}"
        elif isinstance(func, PowerLawFunction):
            return f"f(x) = {params[0]:.4f} * x^{params[1]:.4f} + {params[2]:.4f}"
        elif isinstance(func, PolynomialFunction):
            terms = [f"{params[i]:.4f} * x^{func.degree-i}" for i in range(func.degree+1)]
            return "f(x) = " + " + ".join(terms)
        elif isinstance(func, LogisticFunction):
            return f"f(x) = {params[3]:.4f} + ({params[0]:.4f} - {params[3]:.4f}) / (1 + exp(-{params[1]:.4f} * (x - {params[2]:.4f})))"
        elif isinstance(func, HyperbolicTangentFunction):
            return f"f(x) = {params[0]:.4f} * tanh({params[1]:.4f} * (x - {params[2]:.4f})) + {params[3]:.4f}"
        else:
            return "Unknown function"

    def plot_results(self):
        if self.data is None or self.best_func is None or self.optimized_params is None:
            return

        x, y = self.data[:2]

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        plot_results(x, y, self.best_func, self.optimized_params, ax=ax)
        self.canvas.draw()

        self.residual_figure.clear()
        ax_residual = self.residual_figure.add_subplot(111)
        plot_residuals(x, y, self.best_func, self.optimized_params, ax=ax_residual)
        self.residual_canvas.draw()

        self.right_panel.setCurrentIndex(0)  # Switch to the Results Plot tab

def main():
    app = QApplication(sys.argv)
    gui = OptimizationGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
