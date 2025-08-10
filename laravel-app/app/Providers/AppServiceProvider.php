<?php

namespace App\Providers;

use App\Models\Scan;
use App\Models\ScanDefect;
use App\Models\ScanThreat;
use App\Observers\ScanObserver;
use Illuminate\Support\ServiceProvider;

class AppServiceProvider extends ServiceProvider
{
    /**
     * Register any application services.
     */
    public function register(): void
    {
        //
    }

    /**
     * Bootstrap any application services.
     */
    public function boot(): void
    {
        //
    }
}
