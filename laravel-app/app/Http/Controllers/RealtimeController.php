<?php

namespace App\Http\Controllers;

use Inertia\Inertia;
use Illuminate\Http\Request;
use App\Models\RealtimeSession;

class RealtimeController extends Controller
{
    // list all realtime sessions
    public function index()
    {
        //
    }

    // page for realtime detection
    public function create()
    {
        //
        return Inertia::render('RealTimeAnalysis/Index', []);
    }

    // analyze uploaded images in realtime
    public function store(Request $request)
    {
        //

    }

    // details of a specific realtime session
    public function show(RealtimeSession $realtimeSession)
    {
        //
    }

    // delete a specific realtime session
    public function destroy(RealtimeSession $realtimeSession)
    {
        //
    }
}
