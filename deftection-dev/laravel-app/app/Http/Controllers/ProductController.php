<?php

namespace App\Http\Controllers;

use Inertia\Inertia;
use App\Models\Product;
use Illuminate\Support\Str;
use Illuminate\Http\Request;
use Illuminate\Validation\Rule;
use Illuminate\Support\Facades\Log;
use App\Services\ProductQueryService;

class ProductController extends Controller
{
    public function __construct(
        protected ProductQueryService $queryService
    ) {}


    public function index(Request $request)
    {
        try {
            $filters = $this->queryService->filters($request);
            $products = $this->queryService->getAllProducts($filters);
            $initialChecksum = $this->queryService->getChecksumAllProducts();

            return Inertia::render(
                'Database/Product/Index',
                [
                    'products' => $products->items(),
                    'filters' => [
                        "current" => $filters,
                        'options' => [
                            'perPageOptions' => [5, 10, 25, 50, 100],
                        ]
                    ],
                    'meta' => [
                        'total' => $products->total(),
                        'per_page' => $products->perPage(),
                        'current_page' => $products->currentPage(),
                        'last_page' => $products->lastPage(),
                        'from' => $products->firstItem(),
                        'to' => $products->lastItem(),
                    ],
                    'initialChecksum' => $initialChecksum
                ]
            );
        } catch (\Exception $e) {
            Log::error('Error fetching products: ' . $e->getMessage(), [
                'user_id' => auth()->id(),
            ]);
            return Inertia::render(
                'Database/Product/Index',
                [
                    'products' => [],
                    'filters' => [
                        "current" => [],
                        'options' => [
                            'perPageOptions' => [5, 10, 25, 50, 100],
                        ]
                    ],
                    'meta' => [
                        'total' => 0,
                        'per_page' => 0,
                        'current_page' => 0,
                        'last_page' => 0,
                        'from' => 0,
                        'to' => 0,
                    ],
                    'initialChecksum' => ''
                ]
            );
        }
    }

    public function indexCheck(Request $request)
    {
        try {
            $lastChecksum = $request->input('checksum', '');
            $currentChecksum = $this->queryService->getChecksumAllProducts();
            $hasChanges = $lastChecksum !== $currentChecksum;

            return response()->json([
                'has_changes' => $hasChanges,
                'checksum' => $currentChecksum,
                'timestamp' => now()->toISOString(),
            ]);
        } catch (\Exception $e) {
            Log::error('Error checking updates: ' . $e->getMessage(), [
                'user_id' => auth()->id(),
                'last_checksum' => $request->input('checksum', ''),
            ]);
            return response()->json([
                'error' => 'Failed to check updates',
                'timestamp' => now()->toISOString()
            ], 500);
        }
    }

    // data for polling
    public function indexApi(Request $request)
    {
        try {
            $filters = $this->queryService->filters($request);
            $product = $this->queryService->getAllProducts($filters);
            $currentChecksum = $this->queryService->getChecksumAllProducts();

            $data = [
                'products' => $product->items(),
                'filters' => [
                    'current' => $filters,
                    'options' => [
                        'perPageOptions' => [5, 10, 25, 50, 100],
                    ]
                ],
                'meta' => [
                    'total' => $product->total(),
                    'per_page' => $product->perPage(),
                    'current_page' => $product->currentPage(),
                    'last_page' => $product->lastPage(),
                    'from' => $product->firstItem(),
                    'to' => $product->lastItem(),
                ],
                'checksum' => $currentChecksum,
                'timestamp' => now()->toISOString(),
            ];

            return response()->json($data);
        } catch (\Exception $e) {
            Log::error('Error in poll data: ' . $e->getMessage(), [
                'filters' => $request->all(),
            ]);
            return response()->json([
                'user_id' => auth()->id(),
                'error' => 'Failed to fetch data',
                'timestamp' => now()->toISOString()
            ], 500);
        }
    }

    public function indexRefresh(Request $request)
    {
        try {
            // This is essentially the same as pollData but ensures fresh data
            return $this->indexApi($request);
        } catch (\Exception $e) {
            Log::error('Error in force refresh: ' . $e->getMessage());
            return response()->json(['error' => 'Failed to refresh data'], 500);
        }
    }

    public function store(Request $request)
    {
        $request->validate([
            'name' => ['required', 'string', 'max:255', 'unique:products'],
        ]);

        $product = Product::create([
            'name' => $request->name,
            'slug' => Str::slug($request->name),
        ]);

        // Log successful product creation
        Log::info('Product created', [
            'product_id' => $product->id,
            'name' => $product->name,
            'slug' => $product->slug,
            'user_id' => auth()->id()
        ]);

        return redirect()
            ->back()
            ->with('success', 'Product created successfully.');
    }

    public function showApi(Product $product)
    {
        try {
            return response()->json([
                'success' => true,
                'message' => 'Product retrieved successfully',
                'data' => $product
            ]);
        } catch (\Exception $e) {
            Log::error('Error showing product: ' . $e->getMessage(), [
                'product_id' => $product->id,
                'user_id' => auth()->id(),
            ]);
            return response()->json([
                'success' => false,
                'message' => 'Failed to show product',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, Product $product)
    {
        $request->validate([
            'name' => [
                'required',
                'string',
                'max:255',
                Rule::unique('products', 'name')->ignore($product->slug, 'slug'),
            ],
        ]);

        // Store original values for logging
        $originalName = $product->name;
        $originalSlug = $product->slug;

        $product->update([
            'name' => $request->name,
            'slug' => Str::slug($request->name),
        ]);

        // Log successful product update
        Log::info('Product updated', [
            'product_id' => $product->id,
            'old_name' => $originalName,
            'new_name' => $product->name,
            'old_slug' => $originalSlug,
            'new_slug' => $product->slug,
            'user_id' => auth()->id()
        ]);


        return redirect()
            ->back()
            ->with('success', 'Product updated successfully.');
    }

    public function destroy(Product $product)
    {
        // Store product details for logging before deletion
        $productData = [
            'product_id' => $product->id,
            'name' => $product->name,
            'slug' => $product->slug,
            'user_id' => auth()->id()
        ];

        $product->delete();

        // Log successful product deletion
        Log::info('Product deleted', $productData);

        return redirect()
            ->back()
            ->with('success', 'Product deleted successfully.');
    }
}
