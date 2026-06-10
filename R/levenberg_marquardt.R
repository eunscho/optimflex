# ============================================================
# Function: levenberg_marquardt
# Description: Optimized LM for Least-Squares/SEM problems.
# Features: fast_jac, max_time, Predicted Decrease (AND rule),
#           and Hessian output.
#   - control$diff_method ("forward" / "central" / "richardson")
#   - Interface (input, output, convergence) synced with gauss_newton()
#   - All internal quantities (gradient, Hessian approx, accept/reject)
#     operate on the OBJECTIVE (F_ML) scale, consistent with the
#     gradient-based convergence test. This matches gauss_newton().
# ============================================================
#' Levenberg-Marquardt Optimization
#'
#' @description
#' Implements a full-featured Levenberg-Marquardt algorithm for non-linear 
#' optimization, specifically optimized for Structural Equation Modeling (SEM).
#'
#' @details
#' \code{levenberg_marquardt} is a specialized optimization algorithm for 
#' least-squares and Maximum Likelihood problems where the objective function 
#' can be expressed as a sum of squared residuals.
#'
#' \bold{Scaling and SEM Consistency:}
#' To ensure consistent simulation results and standard error (SE) calculations,
#' this implementation adjusts the Gradient \eqn{(2J^T r)} and the Approximate
#' Hessian \eqn{(2J^T J)} to match the scale of the Maximum Likelihood (ML)
#' fitting function \eqn{F_{ML}}. The damped Newton step, the gain ratio, and the
#' gradient-based convergence test all operate on this same scale so that the
#' algorithm minimizes the actual objective rather than a residual surrogate.
#'
#' \bold{Comparison with Gauss-Newton:}
#' Unlike \code{gauss_newton}, which uses a backtracking line search (Armijo),
#' Levenberg-Marquardt controls step size via a damping parameter \eqn{\lambda}
#' that interpolates between Gauss-Newton and gradient descent. The gain ratio
#' (actual vs. predicted decrease) governs \eqn{\lambda} adjustment following
#' Nielsen's (1999) update rule.
#'
#' @references
#' \itemize{
#'   \item Levenberg, K. (1944). A method for the solution of certain non-linear
#'     problems in least squares. \emph{Quarterly of Applied Mathematics}, \bold{2}(2),
#'     164--168. \doi{10.1090/qam/10666}
#'
#'   \item Marquardt, D. W. (1963). An algorithm for least-squares estimation of
#'     nonlinear parameters. \emph{Journal of the Society for Industrial and Applied
#'     Mathematics}, \bold{11}(2), 431--441. \doi{10.1137/0111030}
#'
#'   \item Nielsen, H. B. (1999). \emph{Damping parameter in Marquardt's method}
#'     (Technical Report IMM-REP-1999-05). Technical University of Denmark.
#'
#'   \item Nocedal, J., & Wright, S. J. (2006). \emph{Numerical Optimization}
#'     (2nd ed.). Springer. \doi{10.1007/978-0-387-40065-5}
#' }
#'
#' @param start Numeric vector. Starting values for the optimization parameters.
#' @param objective Function. The objective function to minimize.
#' @param residual Function (optional). Function that returns the residuals vector.
#' @param gradient Function (optional). Gradient of the objective function.
#' @param hessian Function (optional). Hessian matrix of the objective function.
#' @param jac Function (optional). Jacobian matrix of the residuals.
#' @param lower Numeric vector. Lower bounds for box constraints.
#' @param upper Numeric vector. Upper bounds for box constraints.
#' @param control List. Control parameters including convergence flags:
#'    \itemize{
#'      \item \code{use_abs_f}: Logical. Use absolute change in objective for convergence.
#'      \item \code{use_rel_f}: Logical. Use relative change in objective for convergence.
#'      \item \code{use_abs_x}: Logical. Use absolute change in parameters for convergence.
#'      \item \code{use_rel_x}: Logical. Use relative change in parameters for convergence.
#'      \item \code{use_grad}: Logical. Use gradient norm for convergence.
#'      \item \code{use_posdef}: Logical. Verify positive definiteness at convergence.
#'      \item \code{use_pred_f}: Logical. Record predicted objective decrease.
#'      \item \code{use_pred_f_avg}: Logical. Record average predicted decrease.
#'      \item \code{diff_method}: String. Method for numerical differentiation.
#'      \item \code{lambda_init}: Numeric. Initial damping parameter (LM-specific).
#'      \item \code{lambda_factor}: Numeric. Multiplier for lambda on failure (LM-specific).
#'      \item \code{lambda_max}: Numeric. Upper limit for lambda (LM-specific).
#'      \item \code{max_time}: Numeric. Time limit in seconds (LM-specific).
#'    }
#' @param ... Additional arguments passed to objective, residual, gradient, 
#'   hessian, and jac functions.
#'
#' @return A list containing optimization results and iteration metadata.
#' @export
#' @examples
#' # Simple quadratic function optimization
#' quad <- function(x) (x[1] - 2)^2 + (x[2] + 1)^2
#' quad_res <- function(x) c(x[1] - 2, x[2] + 1)
#' res <- levenberg_marquardt(start = c(0, 0), objective = quad, residual = quad_res)
#' print(res$par)
levenberg_marquardt <- function(
    start,
    objective,
    residual       = NULL,
    gradient       = NULL,
    hessian        = NULL,
    jac            = NULL,
    lower          = -Inf,
    upper          = Inf,
    control        = list(),
    ...
) {
  
  # ---------- 1. Default Configuration (Synced with Suite) ----------
  ctrl0 <- list(
    # Convergence and recording flags
    use_abs_f       = FALSE,
    use_rel_f       = FALSE,
    use_abs_x       = FALSE,
    use_rel_x       = TRUE,
    use_grad        = TRUE,
    use_posdef      = TRUE,
    use_pred_f      = FALSE,
    use_pred_f_avg  = FALSE,
    
    # Algorithm parameters
    max_iter        = 1000L,
    tol_abs_f       = 1e-6,
    tol_rel_f       = 1e-6,
    tol_abs_x       = 1e-6,
    tol_rel_x       = 1e-6,
    tol_grad        = 1e-4,
    tol_pred_f      = 1e-4,
    tol_pred_f_avg  = 1e-4,
    diff_method     = "forward",
    
    # LM-specific parameters
    max_time        = Inf,
    lambda_init     = 1e-3,
    lambda_factor   = 10.0,
    lambda_max      = 1e12,
    eps             = 1e-5
  )
  ctrl <- utils::modifyList(ctrl0, control)
  ctrl$diff_method <- match.arg(ctrl$diff_method, c("forward", "central", "richardson"))
  
  if (ctrl$diff_method == "richardson") {
    if (!requireNamespace("numDeriv", quietly = TRUE)) stop("Package 'numDeriv' required.")
  }
  
  # ---------- 2. Internal Helpers (Synced with Suite) ----------
  eval_obj <- function(z) as.numeric(objective(z, ...))[1]
  
  grad_func <- if (!is.null(gradient)) {
    function(z) as.numeric(gradient(z, ...))
  } else if (ctrl$diff_method == "richardson") {
    function(z) as.numeric(numDeriv::grad(objective, z, method = "Richardson", ...))
  } else {
    function(z) fast_grad(objective, z, diff_method = ctrl$diff_method, ...)
  }
  
  hess_func <- if (!is.null(hessian)) {
    function(z) hessian(z, ...)
  } else if (ctrl$diff_method == "richardson") {
    function(z) numDeriv::hessian(objective, z, method = "Richardson", ...)
  } else {
    function(z) fast_hess(objective, z, diff_method = ctrl$diff_method, ...)
  }
  
  jac_func <- if (!is.null(jac)) {
    function(z) jac(z, ...)
  } else if (!is.null(residual)) {
    if (ctrl$diff_method == "richardson") {
      function(z) numDeriv::jacobian(residual, z, method = "Richardson", ...)
    } else {
      function(z) fast_jac(residual, z, diff_method = ctrl$diff_method, ...)
    }
  } else {
    NULL
  }
  
  # Gradient on the objective (F_ML) scale.
  # If an analytic gradient is supplied, use it directly; otherwise fall back
  # to 2*J'r (the gradient of sum(r^2)), matching gauss_newton().
  get_g <- function(curr_x, curr_J) {
    if (!is.null(gradient)) return(grad_func(curr_x))
    if (!is.null(residual)) return(as.numeric(2 * crossprod(curr_J, as.numeric(residual(curr_x, ...)))))
    return(grad_func(curr_x))
  }
  
  # ---------- 3. Initialization ----------
  param_names <- names(start)
  x <- as.numeric(start); n_par <- length(x)
  start_clock <- proc.time()
  
  f <- tryCatch(eval_obj(x), error = function(e) NA_real_)
  it <- 0L; x_old <- x; f_old <- NA_real_; converged <- FALSE; status <- "running"
  H_curr <- NULL; H_eval <- NULL; g_inf <- NA_real_; pred_dec <- NA_real_; pred_dec_avg <- NA_real_
  
  if (!is.finite(f)) {
    status <- "objective_error_at_start"
  } else if (is.null(jac_func)) {
    status <- "jacobian_unavailable"
  } else {
    J <- jac_func(x)
    if (is.null(J) || any(!is.finite(J))) {
      status <- "jacobian_error_at_start"
    } else {
      # All quantities on the objective (F_ML) scale:
      #   g      = grad of objective  (analytic, or 2*J'r)
      #   H_curr = 2*J'J  (Gauss-Newton approximation of the objective Hessian)
      g <- get_g(x, J)
      H_curr <- 2 * crossprod(J)
      
      lambda <- ctrl$lambda_init
      
      # ---------- 4. Main Optimization Loop ----------
      tryCatch({
        repeat {
          # Time limit check
          if (as.numeric((proc.time() - start_clock)[3]) >= ctrl$max_time) { status <- "time_limit_reached"; break }
          if (it >= ctrl$max_iter) { status <- "iteration_limit_reached"; break }
          
          it <- it + 1L
          g_inf <- max(abs(g), na.rm = TRUE)
          
          # 4.1) Damped Newton (LM) Step on the objective scale:
          #      (2*J'J + lambda*I) p = -g
          p_step <- tryCatch({
            H_mod <- H_curr
            diag(H_mod) <- diag(H_mod) + lambda
            solve(H_mod, -g)
          }, error = function(e) NULL)
          
          if (is.null(p_step)) {
            lambda <- lambda * ctrl$lambda_factor
            if (lambda > ctrl$lambda_max) { status <- "singular_matrix_fail"; break }
            next
          }
          
          # 4.2) Predicted Decrease (objective scale): -(g'p + 0.5 p'H p)
          Hp <- as.numeric(H_curr %*% p_step)
          pred_dec <- as.numeric(-(sum(g * p_step) + 0.5 * sum(p_step * Hp)))
          pred_dec_avg <- pred_dec / n_par
          
          # 4.3) Convergence Verification (matching gauss_newton)
          res_conv <- TRUE
          if (ctrl$use_grad)                     res_conv <- res_conv && (g_inf <= ctrl$tol_grad)
          if (ctrl$use_abs_f && !is.na(f_old))   res_conv <- res_conv && (abs(f - f_old) <= ctrl$tol_abs_f)
          if (ctrl$use_rel_f && !is.na(f_old))   res_conv <- res_conv && (abs((f - f_old) / max(1, abs(f_old))) <= ctrl$tol_rel_f)
          if (ctrl$use_abs_x && it > 1L)         res_conv <- res_conv && (max(abs(x - x_old)) <= ctrl$tol_abs_x)
          if (ctrl$use_rel_x && it > 1L)         res_conv <- res_conv && (max(abs(x - x_old)) / max(1, max(abs(x_old)))) <= ctrl$tol_rel_x
          if (isTRUE(ctrl$use_pred_f))           res_conv <- res_conv && (is.finite(pred_dec) && pred_dec <= ctrl$tol_pred_f)
          if (isTRUE(ctrl$use_pred_f_avg))       res_conv <- res_conv && (is.finite(pred_dec_avg) && pred_dec_avg <= ctrl$tol_pred_f_avg)
          
          if (res_conv && it > 1L) { converged <- TRUE; status <- "converged"; break }
          
          # 4.4) Trial Step and Gain Ratio (objective scale via eval_obj)
          x_try <- x + p_step
          f_try <- tryCatch(eval_obj(x_try), error = function(e) NA_real_)
          
          if (!is.finite(f_try)) {
            lambda <- lambda * ctrl$lambda_factor
            if (lambda > ctrl$lambda_max) { status <- "divergence_lambda_max"; break }
            next
          }
          
          actual_red <- f - f_try
          rho <- if (is.finite(pred_dec) && abs(pred_dec) > 1e-18) actual_red / pred_dec else 0
          
          # 4.5) Accept / Reject Decision
          if (rho > 1e-4 && actual_red > 0) {
            x_old <- x; f_old <- f
            x <- x_try; f <- f_try
            
            # Update Jacobian, gradient, and Hessian approximation
            J <- jac_func(x)
            if (is.null(J) || any(!is.finite(J))) { status <- "jacobian_error"; break }
            g <- get_g(x, J)
            H_curr <- 2 * crossprod(J)
            
            # Nielsen's lambda update
            lambda <- lambda * max(1/3, 1 - (2 * rho - 1)^3)
            lambda <- max(lambda, 1e-9)
            
            # Post-step convergence check (handles exact solutions, e.g., quadratics)
            g_inf_new <- max(abs(g), na.rm = TRUE)
            if (ctrl$use_grad && g_inf_new <= ctrl$tol_grad) {
              g_inf <- g_inf_new
              converged <- TRUE; status <- "converged"; break
            }
          } else {
            lambda <- lambda * ctrl$lambda_factor
            if (lambda > ctrl$lambda_max) { status <- "divergence_lambda_max"; break }
          }
        }
      }, error = function(e) {
        status <<- paste0("runtime_error: ", conditionMessage(e))
      })
    }
  }
  
  # ---------- 5. Finalization & Mandatory PD Check (matching gauss_newton) ----------
  if (!is.null(param_names)) names(x) <- param_names
  
  if (converged) {
    H_eval <- tryCatch(hess_func(x), error = function(e) NULL)
    Hess_pd <- if (!is.null(H_eval)) is_pd_fast(H_eval) else FALSE
    
    if (isTRUE(ctrl$use_posdef) && !Hess_pd) {
      converged <- FALSE
      status <- "converged_but_not_positive_definite"
    }
  } else {
    Hess_pd <- FALSE
  }
  
  H_final <- if (!is.null(H_eval)) H_eval else if (!is.null(H_curr)) H_curr else NA_real_
  final_clock <- proc.time() - start_clock
  
  list(par = x, objective = f, converged = converged, status = status, iter = it,
       cpu_time = as.numeric(final_clock[1] + final_clock[2]),
       elapsed_time = as.numeric(final_clock[3]),
       max_grad = as.numeric(g_inf), Hess_is_pd = Hess_pd,
       Hessian = H_final, approx_hessian = H_curr,
       pred_dec = pred_dec, pred_dec_avg = pred_dec_avg)
}