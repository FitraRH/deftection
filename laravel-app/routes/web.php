<?php

use Inertia\Inertia;
use Illuminate\Support\Facades\Route;
use App\Http\Controllers\ScanController;
use App\Http\Controllers\UserController;
use App\Http\Controllers\ReportController;
use App\Http\Controllers\ProductController;
use App\Http\Controllers\RealtimeController;
use App\Http\Controllers\Auth\LoginController;
use App\Http\Controllers\Auth\RegisterController;
use App\Http\Controllers\DefectTypeController;
use App\Models\DefectType;

Route::permanentRedirect('/', '/login');
Route::post('/logout', [LoginController::class, 'destroy'])->middleware('auth')->name('logout');

Route::middleware(['guest'])->group(function () {
  Route::get('/login', [LoginController::class, 'create'])->name('login');
  Route::post('/login', [LoginController::class, 'store'])->name('login.store');
  Route::get('/register', [RegisterController::class, 'create'])->name('register');
  Route::post('/register', [RegisterController::class, 'store'])->name('register.store');
});

Route::middleware(['auth'])->group(function () {
  //DASHBOARD
  Route::get('/dashboard', function () {
    return Inertia::render('Dashboard/Index');
  })->name('dashboard');

  // PRODUCTS
  Route::prefix('database/products')->group(function () {
    // Product list
    Route::get('/', [ProductController::class, 'index'])->name('products.index');
    Route::get('/api', [ProductController::class, 'indexApi'])->middleware('throttle:60,1')->name('products.index.api');
    Route::get('/check-updates', [ProductController::class, 'indexCheck'])->middleware('throttle:120,1')->name('products.index.check');
    Route::get('/force-refresh', [ProductController::class, 'indexRefresh'])->middleware('throttle:60,1')->name('products.index.refresh');
    Route::get('/{product}', [ProductController::class, 'showApi'])->middleware('throttle:60,1')->name('products.show.api');
    // Product operations
    Route::post('/', [ProductController::class, 'store'])->name('products.store');
    Route::put('/{product}', [ProductController::class, 'update'])->name('products.update');
    Route::delete('/{product}', [ProductController::class, 'destroy'])->name('products.destroy');
  });

  // DEFECT TYPES
  Route::prefix('database/defectTypes')->group(function () {
    // Defect Type list
    Route::get('/', [DefectTypeController::class, 'index'])->name('defect_types.index');
    Route::get('/api', [DefectTypeController::class, 'indexApi'])->middleware('throttle:60,1')->name('defect_types.index.api');
    Route::get('/check-updates', [DefectTypeController::class, 'indexCheck'])->middleware('throttle:120,1')->name('defect_types.index.check');
    Route::get('/force-refresh', [DefectTypeController::class, 'indexRefresh'])->middleware('throttle:60,1')->name('defect_types.index.refresh');
    Route::get('/{defectType}', [DefectTypeController::class, 'showApi'])->middleware('throttle:60,1')->name('defect_types.show.api');
    // Defect Type operations
    Route::post('/', [DefectTypeController::class, 'store'])->name('defect_types.store');
    Route::put('/{defectType}', [DefectTypeController::class, 'update'])->name('defect_types.update');
    Route::delete('/{defectType}', [DefectTypeController::class, 'destroy'])->name('defect_types.destroy');
  });

  // USERS
  Route::prefix('database/users')->group(function () {
    // User list
    Route::get('/', [UserController::class, 'index'])->name('users.index');
    Route::get('/api', [UserController::class, 'indexApi'])->middleware('throttle:60,1')->name('users.index.api');
    Route::get('/check-updates', [UserController::class, 'indexCheck'])->middleware('throttle:120,1')->name('users.index.check');
    Route::get('/force-refresh', [UserController::class, 'indexRefresh'])->middleware('throttle:60,1')->name('users.index.refresh');
    Route::get('/{user}', [UserController::class, 'showApi'])->middleware('throttle:60,1')->name('users.show.api');
    // User operations
    Route::post('/', [UserController::class, 'store'])->name('users.store');
    Route::put('/{user}', [UserController::class, 'update'])->name('users.update');
    Route::delete('/{user}', [UserController::class, 'destroy'])->name('users.destroy');
  });

  // REPORTS
  Route::prefix('reports')->group(function () {
    // Single scan report routes
    Route::get('/scan/{scan}/generate', [ReportController::class, 'generateSingleReport'])
      ->name('reports.single.generate');
    Route::get('/scan/{scan}/preview', [ReportController::class, 'previewSingleReport'])
      ->name('reports.single.preview');

    // Future batch report routes (for later implementation)
    Route::get('/batch/generate', [ReportController::class, 'generateBatchReport'])
      ->name('reports.batch.generate');
    Route::get('/batch/preview', [ReportController::class, 'previewBatchReport'])
      ->name('reports.batch.preview');
  });

  //SETTINGS
  Route::get('/settings', function () {
    return Inertia::render('Settings/Index');
  })->name('settings');



  Route::get('/image-analysis', function () {
    return Inertia::render('ImageAnalysis/Index');
  })->name('image-analysis');


  // SCANS 
  // scans store (image analysis)
  Route::get('/image-analysis', [ScanController::class, 'create'])->name('scans.create');
  Route::post('image-analysis', [ScanController::class, 'store'])->name('scans.store');
  // Scan list (database) (admin only)
  Route::prefix('database/scans')->group(function () {
    //   Route::get('/', [ScanController::class, 'index'])->name('scans.index');  
    //   Route::get('/api', [ScanController::class, 'indexApi'])->middleware('throttle:60,1')->name('scans.index.api');
    //   Route::get('/check-updates', [ScanController::class, 'checkUpdates'])->middleware('throttle:120,1')->name('scans.index.check-updates');
    //   Route::get('/force-refresh', [ScanController::class, 'forceRefresh'])->middleware('throttle:60,1')->name('scans.index.forceRefresh');
    Route::get('/{scan}', [ScanController::class, 'show'])->name('scans.show'); // only current auth user or admin
    //   // Scan list operations
    //   Route::delete('/{scan}', [ScanController::class, 'destroy'])->name('scans.destroy');
  });
  // Scan myList (analysis)
  Route::prefix('analysis/scan-history')->group(function () {
    Route::get('/', [ScanController::class, 'myScans'])->name('scans.myscans');
    Route::get('/api', [ScanController::class, 'myScansApi'])->middleware('throttle:60,1')->name('scans.myscans.api');
    Route::get('/check-updates', [ScanController::class, 'myScansCheck'])->middleware('throttle:120,1')->name('scans.myscans.check');
    Route::get('/force-refresh', [ScanController::class, 'myScansRefresh'])->middleware('throttle:60,1')->name('scans.myscans.refresh');
    // Route::get('/{scan}', [ScanController::class, 'showMyScan'])->name('scans.show-myscan');

    // Scan myList operations
    Route::delete('/{scan}', [ScanController::class, 'destroyMyScan'])->name('scans.destroy-myscan');
  });

  // Route::get('/{scan}', [ScanController::class, 'show'])->name('analysis.show');


  // REALTIME SESSIONS
  // sessions store (realtime analysis)
  Route::get('/realtime-detection', [RealtimeController::class, 'create'])->name('sessions.create');
  // Route::post('realtime-detection', [RealtimeController::class, 'store'])->name('sessions.store');
  // Session list (database) (admin only)
  Route::prefix('database/realtime-sessions')->group(function () {
    //   Route::get('/', [RealtimeController::class, 'index'])->name('sessions.index');
    //   Route::get('/api', [RealtimeController::class, 'indexApi'])->middleware('throttle:60,1')->name('sessions.index.api');
    //   Route::get('/check-updates', [RealtimeController::class, 'checkUpdates'])->middleware('throttle:120,1')->name('sessions.index.check-updates');
    //   Route::get('/force-refresh', [RealtimeController::class, 'forceRefresh'])->middleware('throttle:60,1')->name('sessions.index.forceRefresh');
    //   // Session list operations
  });
  // Session myList
  Route::prefix('analysis/session-history')->group(function () {
    // Route::get('/', [RealtimeController::class, 'mySessions'])->name('sessions.mysessions');
    // Route::get('/api', [RealtimeController::class, 'mySessionsApi'])->middleware('throttle:60,1')->name('sessions.mysessions.api');
    // Route::get('/check-updates', [RealtimeController::class, 'mySessionsCheck'])->middleware('throttle:120,1')->name('sessions.mysessions.check');
    // Route::get('/force-refresh', [RealtimeController::class, 'mySessionsRefresh'])->middleware('throttle:60,1')->name('sessions.mysessions.refresh');

    // Sessions myList operations
  });
});

Route::get('/home', function () {
  return Inertia::render('Database/ProductOld');
});
