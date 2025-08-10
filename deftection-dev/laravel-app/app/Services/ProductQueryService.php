<?php

namespace App\Services;

use App\Models\Product;
use Illuminate\Http\Request;
use Illuminate\Support\Carbon;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;

class ProductQueryService
{
    public function getAllProducts(array $filters): LengthAwarePaginator
    {
        return Product::query()
            ->when(
                $filters['search'],
                fn($query) =>
                $query->where('name', 'like', '%' . $filters['search'] . '%')
            )
            ->when(!empty($filters['dateFrom']), function ($query) use ($filters) {
                $query->whereDate('created_at', '>=', $filters['dateFrom']);
            })
            ->when(!empty($filters['dateTo']), function ($query) use ($filters) {
                $query->whereDate('created_at', '<=', $filters['dateTo']);
            })
            ->orderBy($filters['sortBy'], $filters['sortDir'])
            ->paginate($filters['perPage'], ['*'], 'page', $filters['page'])
            ->withQueryString()
            ->through(fn($product) => [
                'id' => $product->id,
                'name' => $product->name,
                'slug' => $product->slug,
                'created_at' => $product->created_at->format('Y-m-d H:i:s'),
                'updated_at' => $product->updated_at->format('Y-m-d H:i:s'),
            ]);
    }

    public function getChecksumAllProducts(): string
    {
        $productData = Product::selectRaw('
            COUNT(*) as total_products,
            MAX(updated_at) as latest_update
        ')->first();

        $checksumData = [
            'total_products' => $productData->total_products ?? 0,
            'latest_update' => $productData->latest_update
                ? Carbon::parse($productData->latest_update)->timestamp
                : 0,
        ];

        return md5(json_encode($checksumData));
    }

    public function filters(Request $request): array
    {
        $validated = $request->validate([
            'search' => 'nullable|string|max:255',
            'sort_by' => 'nullable|string|in:' . implode(',', Product::$sortable),
            'sort_dir' => 'nullable|string|in:asc,desc',
            'per_page' => 'nullable|integer|min:1|max:100',
            'page' => 'nullable|integer|min:1',
            'date_from' => 'nullable|date',
            'date_to' => 'nullable|date|after_or_equal:date_from',
        ]);
        return [
            'search' => $validated['search'] ?? '',
            'sortBy' => $validated['sort_by'] ?? 'created_at',
            'sortDir' => $validated['sort_dir'] ?? 'desc',
            'perPage' => $validated['per_page'] ?? 10,
            'page' => $validated['page'] ?? 1,
            'status' => $validated['status'] ?? [],
            'dateFrom' => $validated['date_from'] ?? null,
            'dateTo' => $validated['date_to'] ?? null,
        ];
    }
}
