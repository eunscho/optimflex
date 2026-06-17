#' Dogleg Trust-Region Optimization
#'
#' @description
#' Implements the standard Powell's Dogleg Trust-Region algorithm for non-linear optimization.
#'
#' @details
#' This function implements the classic Dogleg method within a Trust-Region framework, 
#' based on the strategy proposed by Powell (1970).
#' 
#' \bold{Trust-Region vs. Line Search:}
#' Trust-Region methods define a neighborhood around the current point (the trust region 
#' with radius \eqn{\Delta}) where a local quadratic model is assumed to be reliable. 
#' Unlike Line Search methods that first determine a search direction and then 
#' find an appropriate step length, this approach constrains the step size first 
#' and then finds the optimal update within that boundary.
#'
#' \bold{Powell's Dogleg Trajectory:}
#' The "Dogleg" trajectory is a piecewise linear path connecting:
#' \enumerate{
#'    \item The current point.
#'    \item The \bold{Cauchy Point} (\eqn{p_C}): The minimizer of the quadratic model along 
#'          the steepest descent direction.
#'    \item The \bold{Newton Point} (\eqn{p_N}): The unconstrained minimizer of the quadratic model \eqn{(B^{-1}g)}.
#' }
#' The algorithm selects a step along this path such that it minimizes the quadratic 
#' model while remaining within the radius \eqn{\Delta}.
#'
#' \bold{Relationship to Double Dogleg:}
#' While the \code{double_dogleg} algorithm (Dennis and Mei, 1979) introduces a bias 
#' point to follow the Newton direction more closely, this standard Dogleg follows 
#' the original two-segment trajectory.
#'
#' @references
#' \itemize{
#'    \item Powell, M. J. D. (1970). A Hybrid Method for Nonlinear Equations. 
#'          \emph{Numerical Methods for Nonlinear Algebraic Equations}.
#'    \item Nocedal, J., & Wright, S. J. (2006). \emph{Numerical Optimization}. Springer.
#' }
#'
#' @param start Numeric vector. Starting values for the optimization parameters.
#' @param objective Function. The objective function to minimize.
#' @param gradient Function (optional). Gradient of the objective function.
#' @param hessian Function (optional). Hessian matrix of the objective function.
#' @param gn_hessian Function (optional). Gauss-Newton curvature (e.g. 2 * t(J) %*% J)
#'   used as the iteration curvature B in place of \code{hessian}, recomputed at every
#'   accepted step (damped Gauss-Newton). When supplied, the positive-definiteness check
#'   at convergence still uses \code{hessian} (the observed Hessian), so a Gauss-Newton
#'   curvature - which is positive definite by construction - does not make that check
#'   vacuous. If \code{hessian} is omitted, the check falls back to finite differences
#'   of the objective.
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
#'    }
#' @param ... Additional arguments passed to objective, gradient, and Hessian functions.
#'
#' @return A list containing optimization results and iteration metadata.
#' @export
#' @examples
#' # Simple quadratic function optimization
#' quad <- function(x) (x[1] - 2)^2 + (x[2] + 1)^2
#' res <- dogleg(start = c(0, 0), objective = quad)
#' print(res$par)
dogleg <- function(
    start, 
    objective, 
    gradient = NULL, 
    hessian  = NULL, 
    gn_hessian = NULL,
    lower    = -Inf, 
    upper    = Inf,
    control  = list(), 
    ...
) {
  # ---------- 1. Configuration ----------
  ctrl0 <- list(
    use_abs_f       = FALSE, 
    use_rel_f       = FALSE, 
    use_abs_x       = FALSE, 
    use_rel_x       = TRUE,
    use_grad        = TRUE, 
    use_posdef      = TRUE,
    use_pred_f      = FALSE,
    use_pred_f_avg  = FALSE,
    
    max_iter        = 10000L, 
    tol_abs_f       = 1e-6, 
    tol_rel_f       = 1e-6, 
    tol_abs_x       = 1e-6, 
    tol_rel_x       = 1e-6,
    tol_grad        = 1e-4, 
    tol_pred_f      = 1e-4,
    tol_pred_f_avg  = 1e-4,
    initial_delta   = 1.0, 
    delta_max       = 100.0, 
    rho_accept      = 0.1, 
    rho_expand      = 0.75,
    delta_shrink    = 0.25, 
    delta_expand    = 2.0, 
    H_init_diag     = 1.0, 
    diff_method      = "forward",
    hessian_update   = "bfgs",
    use_damped      = TRUE, 
    damp_phi        = 0.2
  )
  ctrl <- utils::modifyList(ctrl0, control)
  ctrl$hessian_update <- match.arg(ctrl$hessian_update, c("bfgs", "exact"))
  
  # ---------- 2. Internal Helpers ----------
  eval_obj <- function(z) as.numeric(objective(z, ...))[1]
  
  grad_func <- if (!is.null(gradient)) {
    function(z) as.numeric(gradient(z, ...))
  } else {
    function(z) fast_grad(objective, z, diff_method = ctrl$diff_method, ...)
  }
  
  # Iteration curvature B: prefer an explicit Gauss-Newton curvature when supplied,
  # otherwise the supplied Hessian, otherwise finite differences of the objective.
  hess_func <- if (!is.null(gn_hessian)) {
    function(z) gn_hessian(z, ...)
  } else if (!is.null(hessian)) {
    function(z) hessian(z, ...)
  } else {
    function(z) fast_hess(objective, z, diff_method = ctrl$diff_method, ...)
  }
  
  # Hessian used for the positive-definiteness check. Always the observed Hessian
  # (a supplied analytic Hessian, else finite differences) - never gn_hessian, which is
  # positive definite by construction and would make the check vacuous.
  hess_func_pd <- if (!is.null(hessian)) {
    function(z) hessian(z, ...)
  } else {
    function(z) fast_hess(objective, z, diff_method = ctrl$diff_method, ...)
  }
  
  project <- function(z, l, u) pmax(l, pmin(z, u))
  
  # ---------- 3. Initialization ----------
  x <- project(as.numeric(start), lower, upper)
  n <- length(x)
  
  # start_clock defined here for CPU time calculation
  start_clock <- proc.time() 
  
  f <- tryCatch(eval_obj(x), error = function(e) NA_real_)
  it <- 0L; converged <- FALSE; status <- "running"
  x_old <- x; f_old <- NA_real_; delta <- ctrl$initial_delta
  
  # Initialize Hessian approximation (B)
  # use_exact_hess: an iteration curvature was supplied - either a Gauss-Newton curvature
  #   (gn_hessian) or a Hessian - and is used to initialize B. The positive-definiteness
  #   check at convergence always uses the observed Hessian (hessian / finite differences),
  #   never gn_hessian.
  # update_exact: recompute the iteration curvature exactly at every accepted step. True
  #   when a Gauss-Newton curvature is supplied, or when a Hessian is supplied AND
  #   hessian_update == "exact". Otherwise (the default "bfgs") B is refined with damped
  #   BFGS rank-two updates from the supplied starting Hessian.
  use_exact_hess <- !is.null(gn_hessian) || !is.null(hessian)
  update_exact <- use_exact_hess && (!is.null(gn_hessian) || identical(ctrl$hessian_update, "exact"))
  B <- if (use_exact_hess) {
    B_init <- tryCatch(hess_func(x), error = function(e) diag(ctrl$H_init_diag, n))
    B_init <- 0.5 * (B_init + t(B_init))
    if (!is_pd_fast(B_init)) {
      ev <- eigen(B_init, symmetric = TRUE, only.values = TRUE)$values
      shift <- max(abs(min(ev)) + 1e-6, max(abs(ev)) * 1e-7)
      B_init + diag(shift, n)
    } else B_init
  } else {
    diag(ctrl$H_init_diag, n)
  }
  H_eval <- NULL; g_inf <- NA_real_
  pred_dec <- NA_real_; pred_dec_avg <- NA_real_
  scaled_B <- FALSE   # whether the one-time self-scaling of B has been applied
  
  # ---------- 4. Main Loop ----------
  if (!is.finite(f)) {
    status <- "objective_error_at_start"
  } else {
    g <- grad_func(x)
    
    tryCatch({
      repeat {
        if (it >= ctrl$max_iter) { status <- "iteration_limit_reached"; break }
        it <- it + 1L
        
        # 4.1) Free Variables Identification (for Box Constraints)
        is_free <- !((x <= lower + 1e-10 & g > 0) | (x >= upper - 1e-10 & g < 0))
        free_idx <- which(is_free); nfree <- length(free_idx)
        
        if (nfree > 0L) {
          g_f <- g[free_idx]; g_inf <- max(abs(g_f), na.rm = TRUE)
          B_f <- B[free_idx, free_idx, drop = FALSE]
          
          # 4.2) Subproblem: Newton Point and Cauchy Point
          # Newton point from a single Cholesky factorization of B_f.
          pN_f <- tryCatch({
            R_f <- chol(B_f)
            backsolve(R_f, forwardsolve(t(R_f), -g_f))
          }, error = function(e) {
            ev <- eigen(B_f, symmetric = TRUE, only.values = TRUE)$values
            shift <- max(abs(min(ev)) + 1e-6, max(abs(ev)) * 1e-7)
            R_f <- chol(B_f + diag(shift, nfree))
            backsolve(R_f, forwardsolve(t(R_f), -g_f))
          })
          
          gnorm <- sqrt(sum(g_f^2)); Bg <- as.numeric(B_f %*% g_f); gBg <- sum(g_f * Bg)
          alpha_c <- if (gBg > 1e-15) (gnorm^2) / gBg else delta / max(gnorm, 1e-12)
          pC_f <- -alpha_c * g_f
          
          # 4.3) Interpolate Dogleg Step based on Delta
          nPN <- sqrt(sum(pN_f^2)); nPC <- sqrt(sum(pC_f^2))
          
          if (nPN <= delta) {
            p_f <- pN_f
          } else if (nPC >= delta) {
            p_f <- (delta / nPC) * pC_f
          } else {
            d_v <- (pN_f - pC_f); aa <- sum(d_v^2); bb <- 2 * sum(pC_f * d_v); cc <- sum(pC_f^2) - delta^2
            tau <- (-bb + sqrt(max(0, bb^2 - 4 * aa * cc))) / (2 * (aa + 1e-16))
            p_f <- pC_f + tau * d_v
          }
          current_pred_dec <- as.numeric(-(sum(g_f * p_f) + 0.5 * sum(p_f * (B_f %*% p_f))))
        } else {
          g_inf <- 0; current_pred_dec <- 0
        }
        
        # 4.4) Convergence Check
        # The full set of convergence criteria is combined with an AND rule, matching
        # gauss_newton(). All tests use the current point (before the step is taken);
        # the predicted-decrease tests reuse current_pred_dec computed in 4.3, which is
        # the quadratic-model predicted decrease of the dogleg step on the free subspace.
        pred_dec <- current_pred_dec
        pred_dec_avg <- current_pred_dec / n
        res_conv <- TRUE
        if (ctrl$use_grad) res_conv <- res_conv && (g_inf <= ctrl$tol_grad)
        if (ctrl$use_abs_f && !is.na(f_old)) res_conv <- res_conv && (abs(f - f_old) <= ctrl$tol_abs_f)
        if (ctrl$use_rel_f && !is.na(f_old)) res_conv <- res_conv && (abs((f - f_old) / max(1, abs(f_old))) <= ctrl$tol_rel_f)
        if (ctrl$use_abs_x && it > 1L) res_conv <- res_conv && (max(abs(x - x_old)) <= ctrl$tol_abs_x)
        if (ctrl$use_rel_x && it > 1L) res_conv <- res_conv && (max(abs(x - x_old)) / max(1, max(abs(x_old)))) <= ctrl$tol_rel_x
        if (isTRUE(ctrl$use_pred_f)) res_conv <- res_conv && (is.finite(pred_dec) && pred_dec <= ctrl$tol_pred_f)
        if (isTRUE(ctrl$use_pred_f_avg)) res_conv <- res_conv && (is.finite(pred_dec_avg) && pred_dec_avg <= ctrl$tol_pred_f_avg)
        
        if (res_conv && it > 1L) {
          if (isTRUE(ctrl$use_posdef)) {
            H_eval <- tryCatch(hess_func_pd(x), error = function(e) NULL)
            if (is_pd_fast(H_eval)) { converged <- TRUE; status <- "converged"; break } else res_conv <- FALSE
          } else { converged <- TRUE; status <- "converged"; break }
        }
        
        # 4.5) Step Acceptance & Hessian Update (Damped BFGS)
        p_full <- rep(0, n); if (nfree > 0L) p_full[free_idx] <- p_f
        x_try <- project(x + p_full, lower, upper); f_try <- eval_obj(x_try)
        actual_red <- f - f_try
        rho <- if (is.finite(current_pred_dec) && current_pred_dec > 1e-15) actual_red / current_pred_dec else 0
        
        if (rho > ctrl$rho_accept && actual_red > 0) {
          g_new <- grad_func(x_try); s <- x_try - x; y <- g_new - g
          
          # Self-scale the initial B on the first BFGS update
          # (Nocedal & Wright 2006, eq. 6.20; direct-Hessian form B0 <- (y'y / s'y) I).
          if (!use_exact_hess && !scaled_B) {
            yy0 <- sum(y * y); sy0 <- sum(s * y)
            if (is.finite(yy0) && yy0 > 1e-12 && is.finite(sy0) && sy0 > 1e-12) {
              B <- diag(yy0 / sy0, n)
              scaled_B <- TRUE
            }
          }
          
          # Damped BFGS Update
          Bs <- as.numeric(B %*% s); sBs <- sum(s * Bs); sy <- sum(s * y)
          update_ok <- FALSE; y_star <- y; sy_star <- sy
          if (isTRUE(ctrl$use_damped)) {
            if (sy < ctrl$damp_phi * sBs) {
              theta_damp <- ((1 - ctrl$damp_phi) * sBs) / (sBs - sy)
              y_star <- theta_damp * y + (1 - theta_damp) * Bs; sy_star <- sum(s * y_star)
            }
            if (sy_star > 1e-12) { y <- y_star; sy <- sy_star; update_ok <- TRUE }
          } else { if (sy > 1e-12) update_ok <- TRUE }
          
          if (update_exact) {
            B_new <- tryCatch(hess_func(x_try), error = function(e) B)
            B_new <- 0.5 * (B_new + t(B_new))
            if (!is_pd_fast(B_new)) {
              ev <- eigen(B_new, symmetric = TRUE, only.values = TRUE)$values
              shift <- max(abs(min(ev)) + 1e-6, max(abs(ev)) * 1e-7)
              B <- B_new + diag(shift, n)
            } else {
              B <- B_new
            }
          } else {
            if (update_ok) {
              B <- B - (Bs %*% t(Bs)) / (sBs + 1e-16) + (y %*% t(y)) / (sy + 1e-16)
              B <- 0.5 * (B + t(B))
            }
          }
          
          x_old <- x; f_old <- f; x <- x_try; f <- f_try; g <- g_new
          
          # Post-step convergence check (handles exact solutions, e.g., quadratics)
          g_inf_new <- max(abs(g_new), na.rm = TRUE)
          if (ctrl$use_grad && g_inf_new <= ctrl$tol_grad) {
            g_inf <- g_inf_new
            if (isTRUE(ctrl$use_posdef)) {
              H_eval <- tryCatch(hess_func_pd(x), error = function(e) NULL)
              if (is_pd_fast(H_eval)) { converged <- TRUE; status <- "converged"; break }
            } else { converged <- TRUE; status <- "converged"; break }
          }
          
          if (rho > ctrl$rho_expand) delta <- min(ctrl$delta_max, ctrl$delta_expand * delta)
        } else {
          delta <- ctrl$delta_shrink * delta
          if (delta < 1e-14) { status <- "radius_too_small"; break }
        }
      }
    }, error = function(e) { status <<- paste0("runtime_error: ", conditionMessage(e)) })
  }
  
  # ---------- 5. Final Reporting ----------
  if (is.null(H_eval)) H_eval <- tryCatch(hess_func(x), error = function(e) NULL)
  final_clock <- proc.time() - start_clock
  
  list(
    par              = x, 
    objective        = f, 
    converged        = converged, 
    status           = status, 
    iter             = it, 
    cpu_time         = as.numeric(final_clock[1] + final_clock[2]), 
    elapsed_time     = as.numeric(final_clock[3]), 
    max_grad         = as.numeric(g_inf), 
    Hessian          = H_eval,
    approx_hessian   = B,
    pred_dec         = pred_dec,
    pred_dec_avg     = pred_dec_avg
  )
}