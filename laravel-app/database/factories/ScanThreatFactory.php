<?php

namespace Database\Factories;

use App\Models\Scan;
use Illuminate\Database\Eloquent\Factories\Factory;

/**
 * @extends \Illuminate\Database\Eloquent\Factories\Factory<\App\Models\ScanThreat>
 */
class ScanThreatFactory extends Factory
{
    /**
     * Define the model's default state.
     *
     * @return array<string, mixed>
     */
    public function definition(): array
    {
        return [
            'scan_id' => Scan::factory(),
            'hash' => $this->faker->sha256,
            'status' => $this->faker->randomElement(['clean', 'malicious', 'suspicious']),
            'risk_level' => $this->faker->randomElement(['none', 'low', 'medium', 'high']),
            'flags' => fn() =>
            collect(range(1, rand(3, 5))) // 3 to 5 items
                ->map(
                    fn() =>
                    implode(' ', array_map('trim', $this->faker->words(rand(1, 2))))
                )
                ->all(),

            'details' => fn() => [
                implode(' ', $this->faker->unique()->words(2)) => $this->faker->sentence,
            ],
            'possible_attack' => fn() =>
            collect(range(1, rand(3, 5))) // 3 to 5 items
                ->map(
                    fn() =>
                    implode(' ', array_map('trim', $this->faker->sentences()))
                )
                ->all(),

            'processing_time_ms' => $this->faker->randomFloat(3, 0, 1000),
            'created_at' => now(),
            'updated_at' => now(),
        ];
    }
}
