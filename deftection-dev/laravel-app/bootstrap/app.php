<?php

use Illuminate\Http\Request;
use Illuminate\Foundation\Application;
use Illuminate\Auth\AuthenticationException;
use App\Http\Middleware\HandleInertiaRequests;
use Illuminate\Validation\ValidationException;
use Illuminate\Auth\Access\AuthorizationException;
use Illuminate\Foundation\Configuration\Exceptions;
use Illuminate\Foundation\Configuration\Middleware;
use Illuminate\Database\Eloquent\ModelNotFoundException;
use Symfony\Component\HttpKernel\Exception\NotFoundHttpException;

return Application::configure(basePath: dirname(__DIR__))
    ->withRouting(
        web: __DIR__ . '/../routes/web.php',
        // api: __DIR__ . '/../routes/api.php',
        commands: __DIR__ . '/../routes/console.php',
        health: '/up',
    )
    ->withMiddleware(function (Middleware $middleware) {
        // Middleware for web requests globally
        $middleware->web(append: [HandleInertiaRequests::class]);

        // Middleware for API requests globally
        $middleware->api(append: []);

        // Middleware for guest users
        $middleware->redirectGuestsTo(fn(Request $request) => route('login'));

        // Middleware for authenticated users
        $middleware->redirectUsersTo(fn(Request $request) => route('dashboard'));
    })
    ->withExceptions(function (Exceptions $exceptions) {
        //

        $exceptions->render(function (NotFoundHttpException $e, $request) {
            if (! $request->expectsJson()) return null;

            return response()->json([
                'success' => false,
                'message' => 'Route not found',
            ], 404);
        });

        $exceptions->render(function (ValidationException $e, $request) {
            if (! $request->expectsJson()) return null;

            return response()->json([
                'success' => false,
                'message' => 'Validation error',
                'errors' => $e->errors(),
            ], 422);
        });

        $exceptions->render(function (ModelNotFoundException $e, $request) {
            if (! $request->expectsJson()) return null;

            return response()->json([
                'success' => false,
                'message' => 'Resource not found',
            ], 404);
        });

        $exceptions->render(function (AuthorizationException $e, $request) {
            if (! $request->expectsJson()) return null;

            return response()->json([
                'success' => false,
                'message' => 'Unauthorized',
            ], 403);
        });

        $exceptions->render(function (AuthenticationException $e, $request) {
            if (! $request->expectsJson()) return null;

            return response()->json([
                'success' => false,
                'message' => 'Unauthenticated',
            ], 401);
        });

        $exceptions->render(function (Throwable $e, $request) {
            if (! $request->expectsJson()) return null;

            return response()->json([
                'success' => false,
                'message' => $e->getMessage(),
            ], 500);
        });
    })->create();
