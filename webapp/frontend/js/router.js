// HCI DocTor - Router System (History API based)

class Router {
    constructor() {
        this.routes = {};
        this.currentPath = '';
        this.notFoundHandler = null;

        // Listen to popstate (back/forward buttons)
        window.addEventListener('popstate', () => {
            this.handleRoute(window.location.pathname);
        });

        // Intercept all link clicks
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-link]')) {
                e.preventDefault();
                this.navigateTo(e.target.getAttribute('href'));
            }
        });
    }

    /**
     * Register a route
     * @param {string} path - Route path (e.g., '/community', '/analysis/:type')
     * @param {function} handler - Function to execute when route matches
     */
    addRoute(path, handler) {
        this.routes[path] = handler;
    }

    /**
     * Set 404 handler
     * @param {function} handler - Function to execute when no route matches
     */
    setNotFound(handler) {
        this.notFoundHandler = handler;
    }

    /**
     * Navigate to a new path
     * @param {string} path - Target path
     */
    navigateTo(path) {
        history.pushState(null, null, path);
        this.handleRoute(path);
    }

    /**
     * Handle route matching and execution
     * @param {string} path - Current path
     */
    handleRoute(path) {
        this.currentPath = path;

        // Try exact match first
        if (this.routes[path]) {
            this.routes[path]();
            return;
        }

        // Try pattern matching (e.g., /analysis/:type)
        for (const route in this.routes) {
            const params = this.matchRoute(route, path);
            if (params) {
                this.routes[route](params);
                return;
            }
        }

        // No match found - 404
        if (this.notFoundHandler) {
            this.notFoundHandler();
        }
    }

    /**
     * Match route with parameters
     * @param {string} route - Route pattern (e.g., '/analysis/:type')
     * @param {string} path - Current path (e.g., '/analysis/ct')
     * @returns {object|null} - Matched parameters or null
     */
    matchRoute(route, path) {
        const routeParts = route.split('/');
        const pathParts = path.split('/');

        if (routeParts.length !== pathParts.length) {
            return null;
        }

        const params = {};

        for (let i = 0; i < routeParts.length; i++) {
            if (routeParts[i].startsWith(':')) {
                // Dynamic parameter
                const paramName = routeParts[i].slice(1);
                params[paramName] = pathParts[i];
            } else if (routeParts[i] !== pathParts[i]) {
                // Static part doesn't match
                return null;
            }
        }

        return params;
    }

    /**
     * Initialize router with current path
     */
    init() {
        const path = window.location.pathname || '/';
        this.handleRoute(path);
    }

    /**
     * Get current path
     */
    getCurrentPath() {
        return this.currentPath;
    }

    /**
     * Get query parameters
     */
    getQueryParams() {
        const params = {};
        const searchParams = new URLSearchParams(window.location.search);
        for (const [key, value] of searchParams) {
            params[key] = value;
        }
        return params;
    }
}

// Export router instance
const router = new Router();
