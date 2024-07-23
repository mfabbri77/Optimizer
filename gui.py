import sys
import time
import threading
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QTextEdit, QLabel, QComboBox, 
                             QProgressBar, QStatusBar, QTabWidget, QSplitter, QCheckBox,
                             QMessageBox)  # Add QMessageBox here
from PyQt5.QtGui import QTextOption
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from data_handling import read_data, DataError, save_residuals_to_file
from parametric_functions import (create_function, LinearFunction2D, QuadraticFunction2D, 
                                  ExponentialFunction2D, GaussianFunction2D, SineFunction2D, 
                                  LogarithmicFunction2D, PowerLawFunction2D, PolynomialFunction2D, 
                                  LogisticFunction2D, HyperbolicTangentFunction2D,
                                  LinearFunction3D, QuadraticFunction3D, ExponentialFunction3D, 
                                  GaussianFunction3D, SineFunction3D, LogarithmicFunction3D, 
                                  PowerLawFunction3D, PolynomialFunction3D, LogisticFunction3D, 
                                  HyperbolicTangentFunction3D)
from optimization import select_best_function, advanced_optimization, update_progress
from visualization import plot_results, plot_residuals

class OptimizationThread(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    time_updated = pyqtSignal(float, float)
    optimization_complete = pyqtSignal(object, object)
    error_occurred = pyqtSignal(str)

    def __init__(self, data, function, fast_mode, functions):
        super().__init__()
        self.data = data
        self.function = function
        self.fast_mode = fast_mode
        self.functions = functions
        self.start_time = None

    def run(self):
        self.start_time = time.time()
        if len(self.data) == 2:
            x, y = self.data
            z = None
        elif len(self.data) == 3:
            x, y, z = self.data
        else:
            self.error_occurred.emit("Invalid data dimensions")
            return

        try:
            if self.function == "Auto Select":
                self.status_updated.emit("Selecting best function...")
                best_func, initial_params = select_best_function(x, y, self.functions, z,
                                                 progress_callback=self.progress_updated.emit,
                                                 status_callback=self.status_updated.emit,
                                                 time_callback=self.update_time)
            else:
                best_func = create_function(self.function, z is not None)
                if z is not None:
                    initial_params = best_func.initial_guess(x, y, z)
                else:
                    initial_params = best_func.initial_guess(x, y)
                self.progress_updated.emit(50)
                self.status_updated.emit(f"Selected {best_func.__class__.__name__} function")
            
            self.status_updated.emit("Starting parameter optimization...")
            optimized_params = advanced_optimization(best_func, x, y, z, 
                                                     progress_callback=self.progress_updated.emit,
                                                     status_callback=self.status_updated.emit,
                                                     time_callback=self.update_time,
                                                     fast_mode=self.fast_mode)
            self.status_updated.emit("Optimization completed")
            self.optimization_complete.emit(best_func, optimized_params)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def update_time(self, progress):
        elapsed_time = time.time() - self.start_time
        estimated_total_time = elapsed_time / (progress / 100)
        remaining_time = estimated_total_time - elapsed_time
        self.time_updated.emit(elapsed_time, remaining_time)

class OptimizationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.residuals = None
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
        self.dump_residuals_button = None  # Inizializziamo l'attributo
        self.residuals = None
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
        self.function_names = [
            "Auto Select",
            "Linear", "Linear 3D",
            "Quadratic", "Quadratic 3D",
            "Exponential", "Exponential 3D",
            "Gaussian", "Gaussian 3D",
            "Sine", "Sine 3D",
            "Logarithmic", "Logarithmic 3D",
            "Power Law", "Power Law 3D",
            "Polynomial (degree 2)", "Polynomial 3D (degree 2)",
            "Polynomial (degree 3)", "Polynomial 3D (degree 3)",
            "Logistic", "Logistic 3D",
            "Hyperbolic Tangent", "Hyperbolic Tangent 3D"
        ]
        self.function_combo.addItems(self.function_names)
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
        self.dump_residuals_button = QPushButton("Dump Residuals")
        self.dump_residuals_button.clicked.connect(self.dump_residuals)
        self.dump_residuals_button.setEnabled(False)
        self.left_layout.addWidget(self.dump_residuals_button)
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

    def dump_residuals(self):
        if self.data is None or self.residuals is None:
            QMessageBox.warning(self, "Warning", "No optimization results available.")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Save Residuals", "", "Text Files (*.txt)")
        if not filename:
            return

        x, y = self.data[:2]
        z = self.data[2] if len(self.data) > 2 else None

        save_residuals_to_file(filename, x, y, self.residuals, z)
        QMessageBox.information(self, "Success", f"Residuals saved to {filename}")

    def load_data(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "Text Files (*.txt)")
        if filename:
            try:
                self.data = read_data(filename)
                self.result_text.append(f"Data loaded from {filename}")
                self.result_text.append(f"Data shape: {len(self.data[0])} points, {len(self.data)} dimensions")
                self.status_bar.showMessage("Data loaded successfully")
                
                # Update function list based on data dimensionality
                is_3d = len(self.data) == 3
                self.function_combo.clear()
                self.function_combo.addItem("Auto Select")
                for name in self.function_names[1:]:  # Skip "Auto Select"
                    if name.endswith("3D") == is_3d:
                        self.function_combo.addItem(name)
            except DataError as e:
                self.result_text.append(f"Error loading data: {str(e)}")
                self.status_bar.showMessage("Error loading data")
                QMessageBox.critical(self, "Data Loading Error", str(e))

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

        is_3d = len(self.data) == 3

        function_names = [
            name for name in self.function_names
            if name != "Auto Select" and (name.endswith("3D") == is_3d or name.startswith("Polynomial"))
        ]
        
        try:
            functions = [create_function(name, is_3d) for name in function_names]
        except ValueError as e:
            self.handle_error(f"Error creating functions: {str(e)}")
            return

        self.optimization_thread = OptimizationThread(self.data, selected_function, fast_mode, functions)
        self.optimization_thread.progress_updated.connect(self.update_progress)
        self.optimization_thread.status_updated.connect(self.update_status)
        self.optimization_thread.time_updated.connect(self.update_time)
        self.optimization_thread.optimization_complete.connect(self.optimization_finished)
        self.optimization_thread.error_occurred.connect(self.handle_error)
        self.optimization_thread.start()

    def handle_error(self, error_message):
        self.result_text.append(f"Error during optimization: {error_message}")
        self.status_bar.showMessage("Optimization failed")
        self.optimize_button.setEnabled(True)
        QMessageBox.critical(self, "Optimization Error", error_message)

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

        self.plot_results()  # Questo metodo ora imposterà self.residuals
        self.dump_residuals_button.setEnabled(True)
        self.optimize_button.setEnabled(True)
        self.status_bar.showMessage("Optimization completed")

    def get_function_string(self, func, params):
        if isinstance(func, LinearFunction2D):
            return f"f(x) = {params[0]:.10f}x + {params[1]:.10f}"
        elif isinstance(func, LinearFunction3D):
            return f"f(x,y) = {params[0]:.10f}x + {params[1]:.10f}y + {params[2]:.10f}"
        elif isinstance(func, QuadraticFunction2D):
            return f"f(x) = {params[0]:.10f}x^2 + {params[1]:.10f}x + {params[2]:.10f}"
        elif isinstance(func, QuadraticFunction3D):
            return f"f(x,y) = {params[0]:.10f}x^2 + {params[1]:.10f}y^2 + {params[2]:.10f}xy + {params[3]:.10f}x + {params[4]:.10f}y + {params[5]:.10f}"
        elif isinstance(func, ExponentialFunction2D):
            return f"f(x) = {params[0]:.10f} * exp({params[1]:.10f}x) + {params[2]:.10f}"
        elif isinstance(func, ExponentialFunction3D):
            return f"f(x,y) = {params[0]:.10f} * exp({params[1]:.10f}x + {params[2]:.10f}y) + {params[3]:.10f}"
        elif isinstance(func, PowerLawFunction2D):
            return f"f(x) = {params[0]:.10f} * x^{params[1]:.10f} + {params[2]:.10f}"
        elif isinstance(func, PowerLawFunction3D):
            return f"f(x,y) = {params[0]:.10f} * x^{params[1]:.10f} * y^{params[2]:.10f} + {params[3]:.10f}"
        elif isinstance(func, LogisticFunction2D):
            return f"f(x) = {params[3]:.10f} + ({params[0]:.10f} - {params[3]:.10f}) / (1 + exp(-{params[1]:.10f}(x - {params[2]:.10f})))"
        elif isinstance(func, LogisticFunction3D):
            return f"f(x,y) = {params[6]:.10f} + ({params[0]:.10f} - {params[6]:.10f}) / (1 + exp(-{params[1]:.10f}(x - {params[2]:.10f}) - {params[3]:.10f}(y - {params[4]:.10f})))"
        elif isinstance(func, GaussianFunction2D):
            return f"f(x) = {params[0]:.10f} * exp(-((x - {params[1]:.10f})^2) / (2 * {params[2]:.10f}^2))"
        elif isinstance(func, GaussianFunction3D):
            return f"f(x,y) = {params[0]:.10f} * exp(-((x - {params[1]:.10f})^2 / (2 * {params[2]:.10f}^2) + (y - {params[3]:.10f})^2 / (2 * {params[4]:.10f}^2))) + {params[5]:.10f}"
        elif isinstance(func, SineFunction2D):
            return f"f(x) = {params[0]:.10f} * sin({params[1]:.10f}x + {params[2]:.10f}) + {params[3]:.10f}"
        elif isinstance(func, SineFunction3D):
            return f"f(x,y) = {params[0]:.10f} * sin({params[1]:.10f}x + {params[2]:.10f}) * sin({params[3]:.10f}y + {params[4]:.10f}) + {params[5]:.10f}"
        elif isinstance(func, LogarithmicFunction2D):
            return f"f(x) = {params[0]:.10f} * log({params[1]:.10f}x) + {params[2]:.10f}"
        elif isinstance(func, LogarithmicFunction3D):
            return f"f(x,y) = {params[0]:.10f} * log({params[1]:.10f}x) + {params[2]:.10f} * log({params[3]:.10f}y) + {params[4]:.10f}"
        elif isinstance(func, HyperbolicTangentFunction2D):
            return f"f(x) = {params[0]:.10f} * tanh({params[1]:.10f}(x - {params[2]:.10f})) + {params[3]:.10f}"
        elif isinstance(func, HyperbolicTangentFunction3D):
            return f"f(x,y) = {params[0]:.10f} * tanh({params[1]:.10f}(x - {params[2]:.10f}) + {params[3]:.10f}(y - {params[4]:.10f})) + {params[5]:.10f}"
        # Add more cases for other function types as needed
        else:
            return f"f(x,y) = {func.__class__.__name__} (parameters: {', '.join([f'{p:.10f}' for p in params])})"

    def plot_results(self):
        if self.data is None or self.best_func is None or self.optimized_params is None:
            return

        x, y = self.data[:2]
        z = self.data[2] if len(self.data) > 2 else None

        is_3d = self.best_func.is3D()

        self.figure.clear()
        if is_3d:
            ax = self.figure.add_subplot(111, projection='3d')
        else:
            ax = self.figure.add_subplot(111)

        plot_results(x, y, self.best_func, self.optimized_params, ax=ax)
        self.canvas.draw()

        self.residual_figure.clear()
        if is_3d:
            ax_residual = self.residual_figure.add_subplot(111, projection='3d')
        else:
            ax_residual = self.residual_figure.add_subplot(111)
        
        _, self.residuals = plot_residuals(x, y, z, self.best_func, self.optimized_params, ax=ax_residual, is_3d=is_3d)
        self.residual_canvas.draw()

        self.right_panel.setCurrentIndex(0)  # Switch to the Results Plot tab



def main():
    app = QApplication(sys.argv)
    gui = OptimizationGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
