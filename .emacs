;; #####################################################################
;; 等宽测试|
;; ABCDefgh|
;; #####################################################################
(set-face-attribute 'default nil :font "YuanMoWen:pixelsize=22")
;(set-face-attribute 'default nil :font "YouYuan:pixelsize=24")


;; #####################################################################
;; `basic'
;; #####################################################################
(global-font-lock-mode t)
(show-paren-mode t)
(setq show-paren-style 'parentheses)

(setq inhibit-startup-message 0)
(tool-bar-mode 0)
(menu-bar-mode 0)
(scroll-bar-mode 0)

(global-linum-mode t)
(setq column-number-mode t)

(setq scroll-margin 3)
(setq scroll-conservatively 10000)
(setq scroll-step 1)
;(setq auto-window-vscroll nil)

(setq blink-cursor-mode nil)
;(setq blink-cursor-delay 1)

(setq frame-title-format "%b")

(setq x-select-enable-clipboard t)  ; shared with clipboard
(setq make-backup-files nil)        ; no ~ file
(add-hook 'before-save-hook 'delete-trailing-whitespace)

(setq ediff-split-window-function 'split-window-horizontally)

(setq ring-bell-function 'ignore)


(prefer-coding-system 'utf-8)
(set-default-coding-systems 'utf-8)
(set-terminal-coding-system 'utf-8)
(set-keyboard-coding-system 'utf-8)


(mapc (lambda (mode)
        (font-lock-add-keywords
         mode
         '(("\\<\\(FIXME\\|TODO\\|NOTE\\)"
	    1 'font-lock-warning-face prepend))))
      '(python-mode org-mode emacs-lisp-mode c-mode))



;; #####################################################################
;; `org-mode'
;; #####################################################################
(add-hook 'org-mode-hook
	  (lambda ()
	    (linum-mode -1)
	    (setq truncate-lines nil)
	    (setq org-src-fontify-natively t)  ; highlight source code
	    (setq org-html-validation-link nil)
	    ))


(defun insert-org-head ()
  (interactive)
  (beginning-of-buffer)
  (princ "#+AUTHOR: wfj
#+EMAIL: wufangjie1223@126.com
#+HTML_HEAD_EXTRA: <style type=\"text/css\"> body {padding-left: 21%;} #table-of-contents {position: fixed; width: 20%; height: 100%; top: 0; left: 0; overflow-x: hidden; overflow-y: scroll;} </style>
#+HTML_MATHJAX: path: \"file:///usr/local/share/MathJax/MathJax.js\"
#+OPTIONS: ^:{} \\n:t email:t
" (current-buffer)))



;; #####################################################################
;; `global-kbd'
;; #####################################################################
; use M-x suspend-frame instead of shortcuts, it's dangerous
(global-unset-key (kbd "C-z"))
(global-unset-key (kbd "C-x C-z"))

;; ;; If you use emacs -nw, M-[ kbd will drive you crazy
;; (global-unset-key (kbd "M-{"))
;; (global-unset-key (kbd "M-}"))
;; (global-set-key (kbd "M-[") 'backward-paragraph)
;; (global-set-key (kbd "M-]") 'forward-paragraph)

(defun recenter-top-bottom-with-redraw ()
  (interactive)
  (recenter-top-bottom)
  (redraw-display))

(if window-system
    (global-set-key (kbd "C-l") 'recenter-top-bottom-with-redraw))



;; #####################################################################
;; `color'
;; useful functions: list-color-display list-faces-display describe-face
;; #####################################################################
(set-foreground-color "white")
(if window-system
    (set-background-color "#131926"))
  ;(set-background-color "black"))


(let ((gray       (if window-system "#666666" "red"))
      (red        (if window-system "#cd3333" "red"))
      (green      (if window-system "#66cd00" "green"))
      (pale-green (if window-system "#9aff9a" "green"))
      (yellow     (if window-system "#ffec8b" "yellow"))
      (gold       (if window-system "#ffb90f" "yellow"))
      (blue       (if window-system "#4682b4" "blue"))
      ;(gray       (if window-system "#666666" "blue"))
      (magenta    (if window-system "#ffa07a" "magenta"))
      (pink       (if window-system "#ffbbff" "magenta"))
      (pale       (if window-system "#d3d3d3" "cyan"))
      (cyan       (if window-system "#00ffff" "cyan"))
      ; NOTE: set the 16 colors on your terminal preference
      )
  (custom-set-faces
   `(font-lock-builtin-face ((t (:foreground ,pink :bold t))) t)
   `(font-lock-comment-face ((t (:foreground ,green))) t)
   `(font-lock-constant-face ((t (:foreground ,gold :bold t))) t)
   `(font-lock-function-name-face ((t (:background ,blue :underline t))) t)
   `(font-lock-keyword-face ((t (:foreground ,cyan :bold t))) t)
   `(font-lock-string-face ((t (:foreground ,magenta))) t)
   `(font-lock-type-face ((t (:foreground ,pale-green :bold t))) t)
   `(font-lock-variable-name-face ((t (:foreground ,yellow))) t)
   `(font-lock-warning-face ((t (:foreground ,pink :bold t))) t)

   `(region ((t (:background ,blue))) t)
   `(org-block ((t (:foreground "white"))) t)
   `(org-table ((t (:background ,gray :foreground "white"))) t)
   `(org-link ((t (:foreground ,cyan :background nil :bold t))) t)
   `(org-formula ((t (:foreground ,magenta :background ,gray))) t)
   ;`(org-table ((t (:foreground ,yellow))) t)
   ;`(org-table ((t (:foreground ,blue))) t)
   `(highlight ((t (:foreground "white" :background ,blue))) t)
   `(lazy-highlight ((t (:foreground "white" :background ,blue))) t)
   `(isearch ((t (:foreground "black" :background ,yellow :underline t))) t)
   `(isearch-fail ((t (:background ,magenta :bold t))) t)
   `(error ((t (:foreground ,red :bold t))) t)
   `(shadow ((t (:foreground ,gray))) t)  ; :bold t

   `(popup-face ((t (:foreground "black" :background ,pale))) t)
   `(popup-summary-face ((t (:foreground ,blue :background ,pale))) t)
   `(popup-tip-face ((t (:foreground "black" :background ,blue))) t)
   '(popup-menu-selection-face ((t (:inherit popup-tip-face))) t)
   '(popup-selection-face ((t (:inherit popup-tip-face))) t)

   '(button ((t (:inherit font-lock-function-name-face))) t)
   '(link ((t (:inherit font-lock-keyword-face :underline t))) t)
   '(link-visited ((t (:inherit font-lock-builtin-face :underline t))) t)
   '(holiday ((t (:inherit button))) t)
   '(diary ((t (:inherit isearch))) t)
   '(minibuffer-prompt ((t (:inherit button))) t)
   '(eshell-prompt ((t (:inherit button))) t)
   '(show-paren-match ((t (:inherit region :bold t))) t)
   '(show-paren-mismatch ((t (:inherit isearch-fail :bold t))) t)
   '(warning ((t (:inherit font-lock-warning-face))) t)
   '(success ((t (:inherit font-lock-type-face))) t)

   `(ediff-even-diff-A ((t (:background ,gray))) t)
   `(ediff-even-diff-Ancestor ((t (:background ,gray))) t)
   `(ediff-even-diff-B ((t (:background ,gray))) t)
   `(ediff-even-diff-C ((t (:background ,gray))) t)
   `(ediff-odd-diff-A ((t (:background ,gray))) t)
   `(ediff-odd-diff-Ancestor ((t (:background ,gray))) t)
   `(ediff-odd-diff-B ((t (:background ,gray))) t)
   `(ediff-odd-diff-C ((t (:background ,gray))) t)
   ;; `(ediff-fine-diff-A ((t (:background ,gray))) t)
   ;; `(ediff-fine-diff-Ancestor ((t (:background ,gray))) t)
   ;; `(ediff-fine-diff-B ((t (:background ,gray))) t)
   ;; `(ediff-fine-diff-C ((t (:background ,gray))) t)
   ;; `(ediff-current-diff-A ((t (:background ,blue))) t)
   ;; `(ediff-current-diff-Ancestor ((t (:background ,blue))) t)
   ;; `(ediff-current-diff-B ((t (:background ,blue))) t)
   ;; `(ediff-current-diff-C ((t (:background ,blue))) t)
   ))



;; #####################################################################
;; `datetime'
;; #####################################################################
(setq display-time-24hr-format t)
(setq display-time-format "%H:%M:%S %m-%d %a")
(setq display-time-interval 1)
(display-time-mode t)

;; (add-hook 'calendar-mode-hook
;; 	  (lambda ()
;; 	    (local-set-key (kbd "q") 'kill-buffer-and-window)
;; 	    (setq calendar-week-start-day 1)
;; 	    ;(setq calendar-mark-holidays-flag t)
;; 	    ))
;; (add-hook 'calendar-today-visible-hook 'calendar-star-date)



;; #####################################################################
;; `eshell'
;; NOTE: use paragraph level move instead of outline-mode
;; #####################################################################
(setq eshell-save-history-on-exit t
      eshell-history-size 4096
      eshell-hist-ignoredups t)

(defvar eshell-histignore
  '("^\\(cd\\|git\\|svn\\|g\\+\\+\\|cc\\|nvcc\\)\\(\\(\\s \\)+.*\\)?$"
    "^("
    "^\\./a\\.out$"
    "^xmodmap ~/\\.xmodmap$"
    "^sudo apt-get \\(update\\|upgrade\\|autoremove\\)$"
    "^man "
    "^\\(sudo \\)?pip[23]? \\(list\\|show\\|search\\)"
    " -\\(-version\\|[Vv]\\)$"
    ))

(setq eshell-input-filter
      #'(lambda (str)
	  (let ((regex eshell-histignore))
	    (not
	     (catch 'break
	       (while regex
		 (if (string-match (pop regex) str)
		     (throw 'break t))))))))


;; #####################################################################
;; `fast-open'
;; #####################################################################
;; ^~-:.@%+=              ; do not change
;; []{}                   ; i think should change
;; $#                     ; emacs change them
;;  \t\n`<>|;!?'"()*\&    ; must change
(defun convert-to-bash-safe-file-name (name)
  (replace-regexp-in-string
   "\\([][ \t\n`<>|;!?'\"()*\\&$#{}]\\)" "\\\\\\1" name))
  ;(replace-regexp-in-string "\\([^a-zA-Z0-9]\\)" "\\\\\\1" name))

(defun fast-open (name)
  (interactive
   (list (read-file-name "Open file: ")))
  (if (file-exists-p name)
      (call-process-shell-command
       (concat "xdg-open " (convert-to-bash-safe-file-name
			    (expand-file-name name))) nil 0)
    (message "Invalid file")))



;; #####################################################################
;; `ibuffer'
;; #####################################################################
(global-set-key (kbd "C-x C-b") 'ibuffer)

(add-hook 'ibuffer-mode-hook
	  (lambda ()
	    (local-set-key (kbd "U") 'ibuffer-unmark-all)
	    ))


;; #####################################################################
;; `dired-mode'
;; #####################################################################
(add-hook 'dired-mode-hook
	  (lambda ()
	    (setq dired-actual-switches "-la")
	    (setq dired-recursive-copies "always")
	    (setq dired-recursive-deletes "always")
	    (local-set-key (kbd "C-RET")
			   (lambda ()
			     (interactive)
			     (let ((name (dired-file-name-at-point)))
			       (if (file-directory-p name)
				   (dired-find-file)
				 (fast-open name)))))
	    (local-set-key (kbd "s")
			   (lambda ()
			     (interactive)
			     (when dired-sort-inhibit
			       (error "Cannot sort this Dired buffer"))
			     (dired-sort-other
			      (read-string "ls switches (must contain -l): "
					   dired-actual-switches))))
	    ))



;; #####################################################################
;; `python'
;; hide / show code, M-x hs-<TAB>
;; #####################################################################
(if (string-equal system-type "gnu/linux")
    (setq python-shell-interpreter "python3"))

(add-hook 'python-mode-hook 'hs-minor-mode)
(add-hook 'inferior-python-mode-hook
	  (lambda ()
	    ;; (outline-minor-mode t)
	    ;; (setq outline-regexp "\\(>>> \\)+")
	    (setq-local paragraph-start "^>>> ")
	    ;(paragraph-separate "")
	    ;(setq comint-use-prompt-regexp t)
	    ;(setq comint-prompt-regexp "^\\(>>> \\)+"
	    ))

(add-hook 'compilation-shell-minor-mode-hook
	  (lambda ()
	    (let ((map compilation-shell-minor-mode-map))
	      (define-key map "\M-{" 'backward-paragraph)
	      (define-key map "\M-}" 'forward-paragraph))))


(setq python-template #s(hash-table test equal))
(dolist
    (pair
     `(("head" . "#!/usr/bin/python3\n# -*- coding: utf-8 -*-\n\n")
       ("main" . "if __name__ == '__main__':\n    ")
       ("path" . ,(concat
		   "import os\n\n\ntry:\n"
		   "    path = os.path.split(os.path.realpath(__file__))[0]\n"
		   "except NameError:\n"
		   "    path = os.getcwd() or os.getenv('PWD')\n\n"))
       ("pdb"  . "import pdb; pdb.set_trace()\n")))
  (puthash (car pair) (cdr pair) python-template))


(defun insert-python-template ()
  (interactive)
  (let* ((type (completing-read "Type wanted: " python-template))
	 (content (gethash type python-template "")))
    (princ content (current-buffer))))



;; #####################################################################
;; `jedi'
;; #####################################################################
(add-to-list 'load-path "~/.emacs.d/el-get/el-get")
(unless (require 'el-get nil 'noerror)
  (with-current-buffer
      (url-retrieve-synchronously
       "https://raw.github.com/dimitri/el-get/master/el-get-install.el")
    (goto-char (point-max))
    (eval-print-last-sexp)))
(el-get 'sync)

(setq jedi:server-args
      '("--sys-path" "/usr/lib/python3/dist-packages"
	"--sys-path" "/usr/local/lib/python3.5/dist-packages"
	"--sys-path" "/usr/lib/python3.5/dist-packages"
	"--sys-path" "/usr/local/lib/python3.6/dist-packages"
	"--sys-path" "/usr/lib/python3.6/dist-packages"
	"--sys-path" "~/packages"))

(setq jedi:setup-keys t)
(setq jedi:complete-on-dot t)
(add-hook 'python-mode-hook 'jedi:setup)
(add-hook 'inferior-python-mode-hook 'jedi:setup)



;; #####################################################################
;; `term-mode'
;; #####################################################################
(defun toggle-term-mode ()
  (interactive)
  (if (term-in-line-mode)
      (term-char-mode)
    (term-line-mode)))

(add-hook 'term-mode-hook
	  (lambda ()
	    (term-line-mode)
	    (local-set-key (kbd "C-c C-j") 'toggle-term-mode)))




;; #####################################################################
;; `utility'
;; #####################################################################
(defun set-line-spacing (n)
  "only work when window-system"
  (interactive
   (list (string-to-int
	  (read-from-minibuffer "Set line-spacing (integer): "))))
  (setq-default line-spacing n))


(defun set-transparency (n)
  "only work when window-system"
  (interactive
   (list (string-to-int (read-from-minibuffer "Set transparency (0~99): "))))
  (let ((n (- 100 (min 99 (max 0 n)))))
    (set-frame-parameter (selected-frame) 'alpha `(,n ,n))))

(set-transparency 10)


(defun exchange-buffer (&optional prefix)
  (interactive "P")
  (if (> (count-windows) 1)
      (let* ((curr (selected-window))
	     (other (if prefix
			(previous-window)
		      (next-window)))
	     (curr-buf (window-buffer))
	     (other-buf (window-buffer other)))
	(set-window-buffer curr other-buf)
	(set-window-buffer other curr-buf)
	;(other-window (if prefix -1 1))
    )))


(defun toggle-split ()
  (interactive)
  (if (= (count-windows) 2)
      (let* ((win1 (window-at 0 0))
	     (curr (selected-window))
	     (win2 (if (eq win1 curr) (next-window) curr))
	     (buf2 (window-buffer win2))
	     (left (car (window-edges win2))))
	(delete-other-windows win1)
	(if (= left 0) (split-window-right) (split-window-below))
	(set-window-buffer (next-window) buf2)
	(if (eq curr win2) (other-window 1)))))

; toggle-truncate-lines


;; #####################################################################
;; `system-specific' `gnu/linux' `darwin' `windows-nt'
;; #####################################################################
(if (string-equal system-type "gnu/linux")
    (call-process-shell-command "xmodmap ~/.xmodmap" nil 0))

(defun clipboard-save ()
  "let terminal save selection to system clipboard"
  (interactive)
  (call-process-region (point-min) (point-max)
		       "xsel" nil 0 nil "--clipboard" "--input"))


;; (let ((i 161))
;;   (while (< i 256)
;;     (if (= 2 (string-width (format "%c" i)))
;; 	(princ (format "%d\t%c|\n" i i)))
;;     (setq i (+ i 1))))

;; ; 对于 org 有用, 但是对于 terminal 的文本逐字显示没用
;; (defun string-width (c) 1)



;; ;; #####################################################################
;; ;; `c-style'
;; ;; #####################################################################
;; (add-hook 'c-mode-hook
;; 	  (lambda ()
;; 	      (c-set-style "ellemtel")
;; 	      (setq c-basic-offset 4)
;; 	      (setq indent-tabs-mode nil)
;;               (local-set-key (kbd "C-c d")
;;                              (lambda ()
;;                                (interactive)
;;                                (manual-entry (current-word))))
;; 	      ))


;; (add-to-list 'load-path "~/.emacs.d/el-get/eim")
;; (autoload 'eim-use-package "eim" "Another emacs input method")
;; (setq eim-use-tooltip nil)
;; (register-input-method
;;  "eim-py" "euc-cn" 'eim-use-package
;;  "拼音" "汉字拼音输入法" "py.txt")


;; (defun char-width-for-chinese (c)
;;   (let ((w (char-width c)))
;;     (cond ((= w 1) (if (or (and (>= c ?\u203b) (< c ?\u2581))
;; 			   (and (>= c ?\u2596) (<= c ?\uffff)))
;; 		       2 1))
;; 	  ((= w 2) (if (or (< c ?\u203b)
;; 			   (and (>= c ?\u2581b) (< c ?\u2596)))
;; 		       1 2))
;; 	  (t w))))

;; (defalias 'char-width-backup (symbol-function 'char-width))
;; (defalias 'string-width-backup (symbol-function 'string-width))


;; (defun char-width (c)
;;   ; for chinese
;;   (cond ((< c 159) (char-width-backup c))
;;   	((< c ?\u3000) 1)
;;   	((< c ?\ufb00) 2)
;;   	(t (char-width-backup c))))

;; (defun string-width2 (c)
;;   (reduce (lambda (x y) (+ x (char-width y)))
;; 	  c
;; 	  :initial-value 0))

;; (defun string-width2 (c)
;;   (reduce '+ (mapcar 'char-width-for-chinese c)))



;; ;;;;; -*- lexical-binding: t -*-

;; (let ((i ?\u3000)
;;       (pre 0))
;;   (while (< i 40000)
;;     (if (= 2 (string-width (format "%c" i)))
;; 	(progn (unless (= (+ pre 1) i)
;; 		 (princ (format "%d\t%c|\n" i i)))
;; 	       (setq pre i)))
;;     (setq i (+ i 1))))


;; from origin org-html-mathjax-template and MathJax sample
(setq org-html-mathjax-template
      "<script type=\"text/x-mathjax-config\">\n    MathJax.Hub.Config({\n        displayAlign: \"%ALIGN\",\n        displayIndent: \"%INDENT\",\n\n        extensions: [\"tex2jax.js\"],\n        jax: [\"input/TeX\",\"output/HTML-CSS\"],\n        tex2jax: {inlineMath: [[\"$\",\"$\"],[\"\\\\(\",\"\\\\)\"]]}\n});\n</script>\n<script type=\"text/javascript\"\n        src=\"%PATH\"></script>")


;; (setq org-html-mathjax-template
;;       "<script type=\"text/x-mathjax-config\">\n    MathJax.Hub.Config({\n        displayAlign: \"%ALIGN\",\n        displayIndent: \"%INDENT\",\n\n        extensions: [\"tex2jax.js\"],\n        jax: [\"input/TeX\",\"output/HTML-CSS\"],\n        tex2jax: {inlineMath: [[\"$\",\"$\"],[\"\\\\(\",\"\\\\)\"]]}\n        TeX: { equationNumbers: {autoNumber: \"%AUTONUMBER\"},\n               MultLineWidth: \"%MULTLINEWIDTH\",\n               TagSide: \"%TAGSIDE\",\n               TagIndent: \"%TAGINDENT\"\n             }\n});\n</script>\n<script type=\"text/javascript\"\n        src=\"%PATH\"></script>")


; tagindent "0em"
